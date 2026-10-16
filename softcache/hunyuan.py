"""SoftCache sampling for HunyuanVideo on a stock diffusers `HunyuanVideoPipeline`."""

import numpy as np
import torch
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps

from .method import Config, decide, euler, execution_device, moved_residual, regrid_tail
from util_seacache import apply_sea_from_ab, rel_l1  # SeaCache submodule, see __init__.py



@torch.no_grad()   # without it the autograd graph keeps the text encoders' GPU weights alive
def encode(pipe, prompt: str):
    """(prompt_embeds, pooled_prompt_embeds, attention_mask) in the transformer's dtype."""
    dtype = pipe.transformer.dtype
    embeds = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=execution_device(pipe, pipe.text_encoder))
    return tuple(t.to(dtype) for t in embeds)


def generate(pipe, prompt, seed: int, cfg: Config, num_steps: int = 50, height: int = 480,
             width: int = 640, num_frames: int = 65, guidance: float = 6.0):
    """One video. `prompt` is a string, or the output of `encode`. Returns (list of PIL frames, stats dict)."""
    tr = pipe.transformer
    embeds = encode(pipe, prompt) if isinstance(prompt, str) else prompt
    device = execution_device(pipe, tr)
    tr.to(device)      # no-op unless diffusers' offload hooks parked it; the loop bypasses them
    prompt_embeds, pooled, attn_mask = (t.to(device) for t in embeds)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = pipe.prepare_latents(1, tr.config.in_channels, height, width, num_frames,
                                   torch.float32, device, generator)
    retrieve_timesteps(pipe.scheduler, num_steps, device,
                       sigmas=np.linspace(1.0, 0.0, num_steps + 1)[:-1])
    sigmas = [float(s) for s in pipe.scheduler.sigmas.cpu()]
    guidance_t = torch.tensor([guidance], dtype=tr.dtype, device=device) * 1000.0

    latents, stats = denoise(tr, latents, sigmas, prompt_embeds, pooled, attn_mask, guidance_t,
                             cfg)
    image = decode(pipe, latents)
    pipe.maybe_free_model_hooks()
    return image, stats


@torch.no_grad()
def denoise(tr, latents, sigmas, prompt_embeds, pooled, attn_mask, guidance, cfg):
    """The sampling loop: per node, measure staleness, decide fresh / cache / jump, predict the
    velocity, and take an Euler step (a jump also shortens the remaining schedule)."""
    p, p_t = tr.config.patch_size, tr.config.patch_size_t
    _, _, frames, height, width = latents.shape
    grid = (frames // p_t, height // p, width // p)
    rotary = tr.rope(latents)
    attention = _attention_mask(grid, attn_mask)

    num_steps = len(sigmas) - 1
    acc = 0.0                       # SeaCache staleness accumulated since the last refresh
    prev_h = None                   # previous node's filtered block-0 input
    r_anchor = r_prev = None        # block residuals of the last two fresh nodes
    sigma_anchor = sigma_prev = None
    actions, n_full, n_cached, n_skipped = [], 0, 0, 0

    i = 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        # sigma * 1000 cast to the running latents' dtype (fp32 at the first node, then the
        # model dtype), as HunyuanVideoPipeline does; no /1000 round trip as in FLUX
        timestep = torch.tensor(sigma * 1000.0, dtype=torch.float32, device=latents.device)
        timestep = timestep.to(latents.dtype).expand(latents.shape[0])
        hs = tr.x_embedder(latents.to(tr.dtype))
        temb, token_replace = tr.time_text_embed(timestep, pooled, guidance)

        # staleness signal: rel-L1 change of the Wiener-filtered, modulated block-0 input
        modulated = tr.transformer_blocks[0].norm1(hs, emb=temb)[0]
        h = _sea_filter(modulated, grid, sigma)
        raw = rel_l1(h, prev_h) if i > 0 else 0.0
        acc += raw
        # at sigma=1 the filter output is ~0, so SeaCache compares against the unfiltered input
        prev_h = (modulated if i == 0 else h).detach()

        action, jf = decide(cfg, i, sigma, raw, acc)
        if action == "fresh":
            acc = 0.0
            enc = tr.context_embedder(prompt_embeds, timestep, attn_mask)
            out = _blocks(tr, hs, enc, temb, attention, rotary, token_replace, grid)
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
        v = _unpatchify(tr.proj_out(tr.norm_out(out, temb)), grid, p, p_t)

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


def _sea_filter(x, grid, sigma):
    """SeaCache's video filter: over the full frames x height x width token grid, power 3."""
    y = apply_sea_from_ab(x.reshape(x.shape[0], *grid, x.shape[-1]), 1.0 - sigma, sigma,
                          power_exp=3.0, dims=(-2, -3, -4), norm_mode="mean")
    return y.reshape(x.shape[0], -1, x.shape[-1])


def _attention_mask(grid, text_mask):
    """Joint video+text mask, as HunyuanVideoTransformer3DModel.forward builds it: every video
    token, then the prompt's valid text tokens; the text padding is masked out."""
    n_video = grid[0] * grid[1] * grid[2]
    seq_len = n_video + text_mask.shape[1]
    valid = n_video + text_mask.sum(dim=1, dtype=torch.int)
    indices = torch.arange(seq_len, device=text_mask.device).unsqueeze(0)
    return (indices < valid.unsqueeze(1)).unsqueeze(1).unsqueeze(1)


def _blocks(tr, hs, enc, temb, attention, rotary, token_replace, grid):
    """Full forward through every transformer block."""
    first_frame_tokens = grid[1] * grid[2]
    for block in list(tr.transformer_blocks) + list(tr.single_transformer_blocks):
        hs, enc = block(hs, enc, temb, attention, rotary, token_replace, first_frame_tokens)
    return hs


def _unpatchify(x, grid, p, p_t):
    """Tokens [B, N, C*p_t*p*p] back to a [B, C, F, H, W] latent."""
    x = x.reshape(x.shape[0], *grid, -1, p_t, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


@torch.no_grad()
def decode(pipe, latents):
    video = pipe.vae.decode(latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor,
                            return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]
