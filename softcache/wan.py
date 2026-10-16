"""SoftCache sampling for Wan2.1-T2V on a stock diffusers `WanPipeline`.

Wan runs real classifier-free guidance: two transformer forwards per node (prompt and negative
prompt). One staleness signal and one decision serve both, since block 0's modulation depends
on the timestep only; each branch keeps its own block residuals for Residual Motion."""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .method import Config, decide, euler, execution_device, moved_residual, regrid_tail
from util_seacache import apply_sea_from_ab, rel_l1  # SeaCache submodule, see __init__.py



@torch.no_grad()   # without it the autograd graph keeps the text encoder's GPU weights alive
def encode(pipe, prompt: str, negative_prompt: str = ""):
    """(prompt_embeds, negative_prompt_embeds) in the transformer's dtype."""
    dtype = pipe.transformer.dtype
    embeds = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,
                                do_classifier_free_guidance=True, device=execution_device(pipe, pipe.text_encoder))
    # clone: the embeds are views into the text encoder's activations, which they would pin
    return tuple(t.to(dtype).clone() for t in embeds)


def generate(pipe, prompt, seed: int, cfg: Config, num_steps: int = 50, height: int = 480,
             width: int = 832, num_frames: int = 81, guidance: float = 6.0):
    """One video. `prompt` is a string, or the output of `encode`. Returns (list of PIL frames, stats dict)."""
    tr = pipe.transformer
    embeds = encode(pipe, prompt) if isinstance(prompt, str) else prompt
    device = execution_device(pipe, tr)
    tr.to(device)      # no-op unless diffusers' offload hooks parked it; the loop bypasses them
    prompt_embeds, negative_embeds = (t.to(device) for t in embeds)
    # Wan ships a UniPC scheduler; SoftCache takes first-order Euler steps on its shifted grid
    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
            shift=pipe.scheduler.config.flow_shift)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    sigmas = [float(s) for s in pipe.scheduler.sigmas.cpu()]
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = pipe.prepare_latents(1, tr.config.in_channels, height, width, num_frames,
                                   torch.float32, device, generator)

    latents, stats = denoise(tr, latents, sigmas, pipe.scheduler.timesteps, prompt_embeds,
                             negative_embeds, guidance, cfg)
    image = decode(pipe, latents)
    pipe.maybe_free_model_hooks()
    return image, stats


@torch.no_grad()
def denoise(tr, latents, sigmas, timesteps, prompt_embeds, negative_embeds, guidance, cfg):
    """The sampling loop: per node, measure staleness, decide fresh / cache / jump, predict the
    guided velocity, and take an Euler step (a jump also shortens the remaining schedule).
    `timesteps` are the scheduler's, one per node of the original `sigmas`."""
    p_t, p_h, p_w = tr.config.patch_size
    _, _, frames, height, width = latents.shape
    grid = (frames // p_t, height // p_h, width // p_w)
    rotary = tr.rope(latents)
    # On the grid, the scheduler's own timestep; a jump's landing is off the grid, and there
    # the timestep is sigma times the grid's timestep/sigma ratio (~1000).
    on_grid = dict(zip(sigmas, timesteps))
    scale = float(timesteps[0]) / sigmas[0]

    num_steps = len(sigmas) - 1
    acc = 0.0                       # SeaCache staleness accumulated since the last refresh
    prev_h = None                   # previous node's filtered block-0 input
    r_anchor = r_prev = None        # (prompt, negative) block residuals of the last two fresh nodes
    sigma_anchor = sigma_prev = None
    actions, n_full, n_cached, n_skipped = [], 0, 0, 0

    i = 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        timestep = on_grid.get(sigma)
        if timestep is None:
            timestep = torch.tensor(sigma * scale, dtype=timesteps.dtype, device=timesteps.device)
        timestep = timestep.expand(latents.shape[0])
        hs = tr.patch_embedding(latents.to(tr.dtype)).flatten(2).transpose(1, 2)
        temb, temb_proj, enc, _ = tr.condition_embedder(timestep, prompt_embeds, None)
        temb_proj = temb_proj.unflatten(1, (6, -1))

        # staleness signal: rel-L1 change of the Wiener-filtered, modulated block-0 input
        modulated = _block0_modulated(tr.blocks[0], hs, temb_proj)
        h = _sea_filter(modulated, grid, sigma)
        raw = rel_l1(h, prev_h) if i > 0 else 0.0
        acc += raw
        # at sigma=1 the filter output is ~0, so SeaCache compares against the unfiltered input
        prev_h = (modulated if i == 0 else h).detach()

        action, jf = decide(cfg, i, sigma, raw, acc)
        if action == "fresh":
            acc = 0.0
            enc_negative = tr.condition_embedder(timestep, negative_embeds, None)[2]
            outs = [_blocks(tr, hs, e, temb_proj, rotary) for e in (enc, enc_negative)]
            if r_anchor is not None:
                r_prev, sigma_prev = r_anchor, sigma_anchor
            r_anchor, sigma_anchor = tuple((out - hs).detach() for out in outs), sigma
            n_full += 1
        else:
            if r_prev is None:      # a single anchor so far: reuse its residuals as they are
                outs = [hs + r for r in r_anchor]
            else:
                outs = [hs + moved_residual(ra, rp, sigma, sigma_anchor, sigma_prev, cfg.beta)
                        for ra, rp in zip(r_anchor, r_prev)]
            n_cached += 1
        v_cond, v_uncond = (_velocity(tr, out, temb, grid) for out in outs)
        v = v_uncond + guidance * (v_cond - v_uncond)

        if action == "jump":
            sigma_next = max(0.0, sigma + jf * (sigma_next - sigma))
            n_before = len(sigmas)
            sigmas = regrid_tail(sigmas, i, sigma_next)
            n_skipped += n_before - len(sigmas)
        latents = euler(latents, sigma_next - sigma, v)
        actions.append(action[0].upper())
        i += 1

    stats = {"actions": "".join(actions), "full": n_full, "cached": n_cached,
             "jumps": actions.count("J"), "skipped_nodes": n_skipped,
             "sigmas": sigmas}
    return latents, stats


def _block0_modulated(block, hs, temb_proj):
    """Block 0's input after its self-attention AdaLN modulation. Wan computes it inline in
    WanTransformerBlock.forward (there is no norm1(hs, emb) module to call), so it is restated."""
    shift, scale, *_ = (block.scale_shift_table + temb_proj.float()).chunk(6, dim=1)
    return (block.norm1(hs.float()) * (1 + scale) + shift).type_as(hs)


def _sea_filter(x, grid, sigma):
    """SeaCache's video filter over the frames x height x width token grid, in 640-channel
    chunks (the filter is per channel, so chunking is exact and caps the FFT's memory)."""
    x5 = x.reshape(x.shape[0], *grid, x.shape[-1])
    y = torch.cat([apply_sea_from_ab(x5[..., c:c + 640], 1.0 - sigma, sigma, power_exp=3.0,
                                     dims=(-2, -3, -4), norm_mode="mean")
                   for c in range(0, x.shape[-1], 640)], dim=-1)
    return y.reshape(x.shape[0], -1, x.shape[-1])


def _blocks(tr, hs, enc, temb_proj, rotary):
    """Full forward through every transformer block."""
    for block in tr.blocks:
        hs = block(hs, enc, temb_proj, rotary)
    return hs


def _velocity(tr, out, temb, grid):
    """Output norm + projection, then tokens back to a [B, C, F, H, W] latent."""
    shift, scale = (tr.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    out = tr.proj_out((tr.norm_out(out.float()) * (1 + scale) + shift).type_as(out))
    p_t, p_h, p_w = tr.config.patch_size
    out = out.reshape(out.shape[0], *grid, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return out.flatten(6, 7).flatten(4, 5).flatten(2, 3)


@torch.no_grad()
def decode(pipe, latents):
    """Un-normalize with the VAE's latent mean/std, as WanPipeline does, then decode."""
    vae = pipe.vae
    x = latents.to(vae.dtype)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(x.device, x.dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(shape).to(x.device, x.dtype)
    video = vae.decode(x / inv_std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]
