"""SoftCache for ComfyUI FLUX: a MODEL patch (per-step fresh/cache) and a SAMPLER (owns the jumps).

The patch hooks ComfyUI's FLUX blocks through `patches_replace`: double block 0 measures staleness
and decides, every later block is skipped on a cached step, and the last single block restores the
cached (Residual-Motion-extrapolated) residual. The sampler sets sigma/config on the shared state,
reads the decision back, and re-spaces the schedule on a jump."""
import torch

from softcache.method import Config, decide, euler, moved_residual, regrid_tail
from util_seacache import apply_sea_from_ab, rel_l1  # SeaCache submodule, see softcache/__init__.py


class State:
    """Shared by the patch (inside the model forward) and the sampler (outside it)."""

    def __deepcopy__(self, memo):   # ModelPatcher.clone deep-copies model_options; keep one object
        return self

    def reset(self, cfg: Config, grid: tuple[int, int]):
        self.cfg, self.grid = cfg, grid
        self.i, self.sigma, self.acc = 0, 1.0, 0.0
        self.prev_h = self.img_in = self.r_anchor = self.r_prev = None
        self.sigma_anchor = self.sigma_prev = None
        self.txt_len = 0
        self.action, self.jf = "fresh", None


def _probe(block, img, vec, grid, sigma):
    """Block-0 modulated input, then its Wiener-filtered version (SeaCache's staleness signal)."""
    mod = block.img_mod(vec)[0]
    modulated = block.img_norm1(img) * (1 + mod.scale) + mod.shift
    x = modulated.reshape(modulated.shape[0], *grid, modulated.shape[-1])
    h = apply_sea_from_ab(x, 1.0 - sigma, sigma, dims=(-2, -3), norm_mode="mean")
    return modulated, h.reshape(modulated.shape)


def _first_block(state, args, extra):
    img = args["img"]
    if img.shape[0] != 1:
        raise ValueError("SoftCache supports batch size 1 with cfg 1 (FLUX.1-dev, guidance-distilled)")
    s = state
    modulated, h = _probe(extra["block"], img, args["vec"], s.grid, s.sigma)
    raw = rel_l1(h, s.prev_h) if s.i > 0 else 0.0
    s.acc += raw
    s.prev_h = (modulated if s.i == 0 else h).detach()   # at sigma=1 the filter output is ~0
    s.action, s.jf = decide(s.cfg, s.i, s.sigma, raw, s.acc)
    s.i += 1
    s.img_in, s.txt_len = img.detach(), args["txt"].shape[1]
    if s.action == "fresh":
        s.acc = 0.0
        return extra["original_block"](args)
    return {"img": img, "txt": args["txt"]}


def _last_block(state, args, extra):
    s = state
    if s.action == "fresh":
        out = extra["original_block"](args)["img"]
        if s.r_anchor is not None:
            s.r_prev, s.sigma_prev = s.r_anchor, s.sigma_anchor
        s.r_anchor, s.sigma_anchor = (out[:, s.txt_len:] - s.img_in).detach(), s.sigma
        return {"img": out}
    if s.r_prev is None:      # a single anchor so far: reuse its residual as is
        r = s.r_anchor
    else:
        r = moved_residual(s.r_anchor, s.r_prev, s.sigma, s.sigma_anchor, s.sigma_prev, s.cfg.beta)
    out = args["img"].clone()
    out[:, s.txt_len:] = s.img_in + r
    return {"img": out}


def run_skip(args):
    return {"img": args["img"], "txt": args.get("txt")}


class SoftCachePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SoftCache"

    def patch(self, model):
        m = model.clone()
        dm = m.model.diffusion_model
        state = State()
        state.reset(Config(), (1, 1))
        for i, block in enumerate(dm.double_blocks):
            if i == 0:
                fn = lambda a, e, b=block: _first_block(state, a, {**e, "block": b})
            else:
                fn = lambda a, e: run_skip(a) if state.action != "fresh" else e["original_block"](a)
            m.set_model_patch_replace(fn, "dit", "double_block", i)
        last = len(dm.single_blocks) - 1
        for i in range(last + 1):
            if i == last:
                fn = lambda a, e: _last_block(state, a, e)
            else:
                fn = lambda a, e: {"img": a["img"]} if state.action != "fresh" else e["original_block"](a)
            m.set_model_patch_replace(fn, "dit", "single_block", i)
        m.model_options["transformer_options"]["softcache"] = state
        return (m,)


@torch.no_grad()   # like ComfyUI's own samplers; otherwise the autograd graph fills the GPU
def sample_softcache(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     tau=0.5, jf_max=8.0, beta=1.0):
    extra_args = {} if extra_args is None else extra_args
    state = extra_args.get("model_options", {}).get("transformer_options", {}).get("softcache")
    if state is None:
        raise ValueError("SoftCache Sampler needs the model to go through the SoftCache Patch node")
    state.reset(Config(tau=tau, jf_max=jf_max, beta=beta), (x.shape[-2] // 2, x.shape[-1] // 2))
    s_in = x.new_ones([x.shape[0]])
    sigmas = [float(s) for s in sigmas]
    i = 0
    while sigmas[i] > 0:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        state.sigma = sigma
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


NODE_CLASS_MAPPINGS = {"SoftCachePatch": SoftCachePatch, "SoftCacheSampler": SoftCacheSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"SoftCachePatch": "SoftCache Patch (FLUX)",
                              "SoftCacheSampler": "SoftCache Sampler"}
