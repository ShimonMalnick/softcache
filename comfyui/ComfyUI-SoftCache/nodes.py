"""SoftCache for ComfyUI FLUX, HunyuanVideo and Wan 2.1: a MODEL patch per model (per-step fresh/cache)
and one SAMPLER (owns the jumps).

Each patch hooks ComfyUI's own block loop through `patches_replace`: block 0 measures staleness and
decides, every later block is skipped on a cached step, and the last block restores the cached
(Residual-Motion-extrapolated) residual. The sampler sets sigma/config on the shared state, reads the
decision back, and re-spaces the schedule on a jump.

Classifier-free guidance (Wan): ComfyUI runs cond and uncond as one batched call or as separate calls,
depending on free memory. Both see the same latent and timestep, so one decision per step serves both
(made by the first call of the step), and each cond keeps its own residuals, keyed by ComfyUI's cond uuid."""
import inspect

import comfy.model_management
import torch

from softcache.method import Config, decide, euler, moved_residual, regrid_tail
from util_seacache import apply_sea_from_ab, rel_l1  # SeaCache, see softcache/__init__.py


class State:
    """Shared by the patch (inside the model forward) and the sampler (outside it)."""

    def __init__(self, probe, token_slice, patch_size):
        self.probe, self.token_slice, self.patch_size = probe, token_slice, patch_size
        self.reset(Config(), (1,) * len(patch_size))

    def __deepcopy__(self, memo):   # ModelPatcher.clone deep-copies model_options; keep one object
        return self

    def reset(self, cfg: Config, grid: tuple[int, ...]):
        self.cfg, self.grid = cfg, grid
        self.step, self.decided = 0, -1     # sampler node index; the node whose decision is current
        self.sigma, self.acc, self.prev_h = 1.0, 0.0, None
        self.anchors = {}                   # cond uuid -> last two fresh (residual, sigma), oldest first
        self.branches = [None]              # cond uuid of each batch row in the current call
        self.img_in = self.tokens = None
        self.action, self.jf, self.actions = "fresh", None, ""


def _first_block(s, args, extra):
    img = args["img"]
    if img.shape[0] != len(s.branches):
        raise ValueError("SoftCache supports batch size 1 (one latent per prompt)")
    if s.decided != s.step:      # first call of this sampler step (cond and uncond may be separate calls)
        modulated, h = s.probe(extra["block"], img[:1], args, s.grid, s.sigma)
        raw = rel_l1(h, s.prev_h) if s.actions else 0.0
        s.acc += raw
        s.prev_h = (modulated if not s.actions else h).detach()   # at sigma=1 the filter output is ~0
        s.action, s.jf = decide(s.cfg, len(s.actions), s.sigma, raw, s.acc)
        s.actions += s.action[0].upper()
        s.decided = s.step
        if s.action == "fresh":
            s.acc = 0.0
    # a copy: newer ComfyUI blocks update their input in place (`img += ...`), which would rewrite an alias
    s.img_in, s.tokens = img.detach().clone(), s.token_slice(args)
    return extra["original_block"](args) if s.action == "fresh" else _skip(args)


def _last_block(s, args, extra):
    if s.action == "fresh":
        out = extra["original_block"](args)["img"]
        residual = (out[:, s.tokens] - s.img_in).detach()
        for b, r in zip(s.branches, residual.split(1)):
            s.anchors[b] = (s.anchors.get(b, []) + [(r, s.sigma)])[-2:]
        return {"img": out}
    rs = []
    for b in s.branches:
        *prev, (r_anchor, sigma_anchor) = s.anchors[b]
        if prev:
            (r_prev, sigma_prev), = prev
            rs.append(moved_residual(r_anchor, r_prev, s.sigma, sigma_anchor, sigma_prev, s.cfg.beta))
        else:                    # a single anchor so far: reuse its residual as is
            rs.append(r_anchor)
    out = args["img"].clone()
    out[:, s.tokens] = s.img_in + torch.cat(rs)
    return {"img": out}


def _skip(args):
    return {k: args[k] for k in ("img", "txt") if k in args}


def _sea(x, grid, sigma, chunk=None, **kw):
    """SeaCache's filter over the token grid, as each release/softcache model file applies it (Wan in
    640-channel chunks: the filter is per channel, so chunking is exact and caps the FFT's memory)."""
    if x.shape[1] != torch.Size(grid).numel():
        raise ValueError(f"SoftCache: {x.shape[1]} image tokens, expected a {grid} grid")
    xg = x.reshape(x.shape[0], *grid, x.shape[-1])
    if chunk is None:   # no torch.cat: its contiguous copy would change rel_l1's summation order
        return apply_sea_from_ab(xg, 1.0 - sigma, sigma, norm_mode="mean", **kw).reshape(x.shape)
    y = torch.cat([apply_sea_from_ab(xg[..., c:c + chunk], 1.0 - sigma, sigma, norm_mode="mean", **kw)
                   for c in range(0, x.shape[-1], chunk)], dim=-1)
    return y.reshape(x.shape)


def _video(x, grid, sigma, chunk=None):
    return _sea(x, grid, sigma, chunk, power_exp=3.0, dims=(-2, -3, -4))


def _probe_flux(block, img, args, grid, sigma):
    """Block-0 modulated image input, then its Wiener-filtered version (SeaCache's staleness signal)."""
    mod = block.img_mod(args["vec"][:1])[0]
    modulated = block.img_norm1(img) * (1 + mod.scale) + mod.shift
    return modulated, _sea(modulated, grid, sigma, dims=(-2, -3))


def _probe_hunyuan(block, img, args, grid, sigma):
    if args.get("modulation_dims_img") is not None:
        raise ValueError("SoftCache supports HunyuanVideo text-to-video (no guiding frame)")
    mod = block.img_mod(args["vec"][:1])[0]
    modulated = block.img_norm1(img) * (1 + mod.scale) + mod.shift
    return modulated, _video(modulated, grid, sigma)


def _probe_wan(block, img, args, grid, sigma):
    """Wan's block-0 self-attention input, modulated in float32 as release/softcache/wan.py does."""
    table = comfy.model_management.cast_to(block.modulation, dtype=torch.float32, device=img.device)
    e = args["vec"][:1].float()
    if e.ndim < 4:      # [B, 6, D]
        shift, scale, *_ = (table + e).chunk(6, dim=1)
    else:               # newer ComfyUI: [B, frames, 6, D], one row per frame's timestep
        if e.shape[1] != 1:
            raise ValueError("SoftCache supports one timestep per step (Wan 2.1 text-to-video)")
        shift, scale, *_ = (table.unsqueeze(0) + e).unbind(2)
    modulated = (block.norm1(img.float()) * (1 + scale) + shift).type_as(img)
    return modulated, _video(modulated, grid, sigma, chunk=640)


def _patch(model, probe, token_slice, patch_size, first_blocks, last_blocks):
    """Hook block 0 of `first_blocks` (probe + decide), skip the rest on a cached step, and restore the
    residual at the last block of `last_blocks`. Each is (patches_replace key, ModuleList)."""
    m = model.clone()
    state = State(probe, token_slice, patch_size)
    loops = [first_blocks] if last_blocks is None else [first_blocks, last_blocks]
    for n, (key, blocks) in enumerate(loops):
        for i, block in enumerate(blocks):
            if n == 0 and i == 0:
                fn = lambda a, e, b=block: _first_block(state, a, {**e, "block": b})
            elif n == len(loops) - 1 and i == len(blocks) - 1:
                fn = lambda a, e: _last_block(state, a, e)
            else:
                fn = lambda a, e: _skip(a) if state.action != "fresh" else e["original_block"](a)
            m.set_model_patch_replace(fn, "dit", key, i)

    previous = m.model_options.get("model_function_wrapper")

    def wrapper(apply_model, args):   # records which cond each batch row is; chains any earlier wrapper
        uuids = args["c"].get("transformer_options", {}).get("uuids")
        state.branches = list(uuids if uuids is not None else args["cond_or_uncond"])
        if previous is not None:
            return previous(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    m.set_model_unet_function_wrapper(wrapper)
    m.model_options["transformer_options"]["softcache"] = state
    return (m,)


class SoftCachePatch:
    """FLUX.1-dev: guidance-distilled, cfg 1. Single blocks run on cat(txt, img)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SoftCache"

    def patch(self, model):
        dm = model.model.diffusion_model
        return _patch(model, _probe_flux, lambda a: slice(a["txt"].shape[1], None), (2, 2),
                      ("double_block", dm.double_blocks), ("single_block", dm.single_blocks))


class SoftCachePatchHunyuan(SoftCachePatch):
    """HunyuanVideo text-to-video: guidance-distilled, cfg 1. Single blocks run on cat(img, txt) up to
    ComfyUI ~2025-04 and on cat(txt, img) since, so the image-token slice follows the installed code."""
    def patch(self, model):
        dm = model.model.diffusion_model
        txt_first = "torch.cat((txt, img), 1)" in inspect.getsource(type(dm).forward_orig)
        tokens = ((lambda a: slice(a["txt"].shape[1], a["txt"].shape[1] + a["img"].shape[1])) if txt_first
                  else (lambda a: slice(0, a["img"].shape[1])))
        return _patch(model, _probe_hunyuan, tokens, tuple(dm.patch_size),
                      ("double_block", dm.double_blocks), ("single_block", dm.single_blocks))


class SoftCachePatchWan(SoftCachePatch):
    """Wan 2.1 text-to-video, with real CFG. One block loop, image tokens only."""
    def patch(self, model):
        dm = model.model.diffusion_model
        return _patch(model, _probe_wan, lambda a: slice(None), tuple(dm.patch_size),
                      ("double_block", dm.blocks), None)


@torch.no_grad()   # like ComfyUI's own samplers; otherwise the autograd graph fills the GPU
def sample_softcache(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     tau=0.5, jf_max=8.0, beta=1.0):
    """Euler on the given sigmas (it replaces Wan's UniPC too), with SoftCache's jumps."""
    extra_args = {} if extra_args is None else extra_args
    state = extra_args.get("model_options", {}).get("transformer_options", {}).get("softcache")
    if state is None:
        raise ValueError("SoftCache Sampler needs the model to go through a SoftCache Patch node")
    grid = tuple(-(-n // p) for n, p in zip(x.shape[2:], state.patch_size))   # ComfyUI pads to the patch
    state.reset(Config(tau=tau, jf_max=jf_max, beta=beta), grid)
    s_in = x.new_ones([x.shape[0]])
    sigmas = [float(s) for s in sigmas]
    i = 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        state.step, state.sigma = i, sigma
        denoised = model(x, sigma * s_in, **extra_args)
        v = (x - denoised) / sigma
        if state.action == "jump":
            sigma_next = max(0.0, sigma + state.jf * (sigma_next - sigma))
            sigmas = regrid_tail(sigmas, i, sigma_next)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        x = euler(x, sigma_next - sigma, v)
        i += 1
    return x


class SoftCacheSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
            "beta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            "jf_max": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "SoftCache"

    def get_sampler(self, tau, beta, jf_max):
        from comfy.samplers import KSAMPLER
        return (KSAMPLER(sample_softcache, {"tau": tau, "beta": beta, "jf_max": jf_max}),)


NODE_CLASS_MAPPINGS = {"SoftCachePatch": SoftCachePatch, "SoftCachePatchHunyuan": SoftCachePatchHunyuan,
                       "SoftCachePatchWan": SoftCachePatchWan, "SoftCacheSampler": SoftCacheSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"SoftCachePatch": "SoftCache Patch (FLUX)",
                              "SoftCachePatchHunyuan": "SoftCache Patch (HunyuanVideo)",
                              "SoftCachePatchWan": "SoftCache Patch (Wan 2.1)",
                              "SoftCacheSampler": "SoftCache Sampler"}
