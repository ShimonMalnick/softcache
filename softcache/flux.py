"""SoftCache sampling for FLUX.1-dev on a stock diffusers `FluxPipeline`."""

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from .method import Config, decide, euler, execution_device, moved_residual, regrid_tail
from util_seacache import apply_sea_from_ab, rel_l1  # SeaCache submodule, see __init__.py



@torch.no_grad()   # without it the autograd graph keeps the text encoders' GPU weights alive
def encode(pipe, prompt: str, max_sequence_length: int = 512):
    """(prompt_embeds, pooled_prompt_embeds, text_ids) on the text encoders' device."""
    return pipe.encode_prompt(prompt=prompt, prompt_2=None, device=execution_device(pipe, pipe.text_encoder),
                              num_images_per_prompt=1, max_sequence_length=max_sequence_length)


def generate(pipe, prompt, seed: int, cfg: Config, num_steps: int = 50, height: int = 1024,
             width: int = 1024, guidance: float = 3.5):
    """One image. `prompt` is a string, or the output of `encode`. Returns (PIL image, stats dict)."""
    tr = pipe.transformer
    embeds = encode(pipe, prompt) if isinstance(prompt, str) else prompt
    device = execution_device(pipe, tr)
    tr.to(device)      # no-op unless diffusers' offload hooks parked it; the loop bypasses them
    prompt_embeds, pooled, text_ids = (t.to(device) for t in embeds)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents, image_ids = pipe.prepare_latents(1, tr.config.in_channels // 4, height, width,
                                              prompt_embeds.dtype, device, generator)
    sc = pipe.scheduler.config
    mu = calculate_shift(latents.shape[1], sc.get("base_image_seq_len", 256),
                         sc.get("max_image_seq_len", 4096), sc.get("base_shift", 0.5),
                         sc.get("max_shift", 1.15))
    retrieve_timesteps(pipe.scheduler, num_steps, device,
                       sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=mu)
    sigmas = [float(s) for s in pipe.scheduler.sigmas.cpu()]
    guidance_t = (torch.full([1], guidance, device=device, dtype=torch.float32)
                  if tr.config.guidance_embeds else None)

    latents, stats = denoise(tr, latents, sigmas, prompt_embeds, pooled, text_ids, image_ids,
                             guidance_t, cfg)
    image = decode(pipe, latents, height, width)
    pipe.maybe_free_model_hooks()
    return image, stats


@torch.no_grad()
def denoise(tr, latents, sigmas, prompt_embeds, pooled, text_ids, image_ids, guidance, cfg):
    """The sampling loop: per node, measure staleness, decide fresh / cache / jump, predict the
    velocity, and take an Euler step (a jump also shortens the remaining schedule)."""
    img_ids = image_ids[0] if image_ids.ndim == 3 else image_ids
    txt_ids = text_ids[0] if text_ids.ndim == 3 else text_ids
    grid_h, grid_w = int(img_ids[:, 1].max().item() + 1), int(img_ids[:, 2].max().item() + 1)
    enc = tr.context_embedder(prompt_embeds)
    rotary = tr.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

    num_steps = len(sigmas) - 1
    acc = 0.0                       # SeaCache staleness accumulated since the last refresh
    prev_h = None                   # previous node's filtered block-0 input
    r_anchor = r_prev = None        # block residuals of the last two fresh nodes
    sigma_anchor = sigma_prev = None
    actions, n_full, n_cached, n_skipped = [], 0, 0, 0

    i = 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        hs, temb = _embed(tr, latents, sigma, pooled, guidance)

        # staleness signal: rel-L1 change of the Wiener-filtered, modulated block-0 input
        modulated = tr.transformer_blocks[0].norm1(hs, emb=temb)[0]
        h = _sea_filter(modulated, grid_h, grid_w, sigma)
        raw = rel_l1(h, prev_h) if i > 0 else 0.0
        acc += raw
        # at sigma=1 the filter output is ~0, so SeaCache compares against the unfiltered input
        prev_h = (modulated if i == 0 else h).detach()

        action, jf = decide(cfg, i, sigma, raw, acc)
        if action == "fresh":
            acc = 0.0
            out = _blocks(tr, hs, enc, temb, rotary)
            if r_anchor is not None:
                r_prev, sigma_prev = r_anchor, sigma_anchor
            r_anchor, sigma_anchor = (out - hs).detach(), sigma
            n_full += 1
        else:
            if r_prev is None:      # a single anchor so far: reuse its residual as is
                out = hs + r_anchor
            else:
                out = hs + moved_residual(r_anchor, r_prev, sigma, sigma_anchor, sigma_prev,
                                          cfg.beta)
            n_cached += 1
        v = tr.proj_out(tr.norm_out(out, temb))

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


def _embed(tr, latents, sigma, pooled, guidance):
    """Token embedding + timestep/guidance embedding, with the scheduler's exact dtype rounding."""
    t = torch.tensor(sigma * 1000.0, dtype=torch.float32, device=latents.device)
    timestep = (t.to(latents.dtype) / 1000).expand(latents.shape[0])
    hs = tr.x_embedder(latents)
    ts = timestep.to(hs.dtype) * 1000
    if guidance is None:
        return hs, tr.time_text_embed(ts, pooled)
    return hs, tr.time_text_embed(ts, guidance.to(hs.dtype) * 1000, pooled)


def _sea_filter(x, grid_h, grid_w, sigma):
    grid = x.reshape(x.shape[0], grid_h, grid_w, x.shape[-1])
    y = apply_sea_from_ab(grid, 1.0 - sigma, sigma, dims=(-2, -3), norm_mode="mean")
    return y.reshape(x.shape[0], -1, x.shape[-1])


def _blocks(tr, hs, enc, temb, rotary):
    """Full forward through every transformer block."""
    for block in list(tr.transformer_blocks) + list(tr.single_transformer_blocks):
        enc, hs = block(hidden_states=hs, encoder_hidden_states=enc, temb=temb,
                        image_rotary_emb=rotary, joint_attention_kwargs=None)
    return hs


@torch.no_grad()
def decode(pipe, latents, height, width):
    x = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    x = x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
