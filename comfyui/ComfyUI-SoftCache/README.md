# ComfyUI-SoftCache

SoftCache for FLUX.1-dev, HunyuanVideo and Wan 2.1 in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
Four nodes:

- **SoftCache Patch (FLUX)**, **SoftCache Patch (HunyuanVideo)**, **SoftCache Patch (Wan 2.1)**: `MODEL` → `MODEL`.
  Decides fresh or cached per step.
- **SoftCache Sampler**: `SAMPLER`. Owns the jumps, so use it instead of Euler (or Wan's UniPC).

## Install

Install **SoftCache** from ComfyUI-Manager (or `comfy node install softcache`) and restart ComfyUI. By hand:

```bash
pip install softcache      # needs only torch; leaves ComfyUI's own packages alone
git clone https://github.com/ShimonMalnick/softcache
cp -r softcache/comfyui/ComfyUI-SoftCache /path/to/ComfyUI/custom_nodes/
```

On first use, SoftCache downloads its pinned SeaCache filter into `~/.cache/softcache` (`SOFTCACHE_HOME` to
move it). Or work from a clone:

```bash
git clone --recursive https://github.com/ShimonMalnick/softcache    # SeaCache is a submodule
ln -s "$PWD/softcache/comfyui/ComfyUI-SoftCache" /path/to/ComfyUI/custom_nodes/ComfyUI-SoftCache
```

Restart ComfyUI.

## Use

Load the model → its **SoftCache Patch** node → `SamplerCustomAdvanced`, with **SoftCache Sampler** as its
sampler and your usual sigmas. For Wan, keep your CFG guider: cond and uncond share one decision per step
and keep their own caches.

| input | default | |
|---|---|---|
| `tau` | 0.5 | higher = faster; `0` = no caching |
| `beta` | 1.0 | 1.0 below ~4x speedup, 0.5 above |
| `jf_max` | 8.0 | max jump length |

Batch size 1, text-to-video only for HunyuanVideo and Wan. Tested on ComfyUI commits `3bfe4e52` (FLUX on GPU) and
`8b25bbc3` (Wan and HunyuanVideo on GPU), all three on CPU at both. Not bit-exact with ComfyUI's Euler sampler at `tau 0` on GPU.
