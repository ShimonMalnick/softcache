# ComfyUI-SoftCache

SoftCache for FLUX.1-dev in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Two nodes:

- **SoftCache Patch (FLUX)**: `MODEL` → `MODEL`. Decides fresh or cached per step.
- **SoftCache Sampler**: `SAMPLER`. Owns the jumps, so use it instead of Euler.

## Install

```bash
git clone --recursive https://github.com/ShimonMalnick/softcache    # SeaCache is a submodule
ln -s "$PWD/softcache/comfyui/ComfyUI-SoftCache" /path/to/ComfyUI/custom_nodes/ComfyUI-SoftCache
```

The nodes import the `softcache` package and SeaCache from the cloned repo, so keep the clone in
place. Restart ComfyUI.

## Use

Load FLUX → **SoftCache Patch (FLUX)** → `SamplerCustomAdvanced`, with **SoftCache Sampler** as its
sampler and your usual sigmas.

| input | default | |
|---|---|---|
| `tau` | 0.5 | higher = faster; `0` = no caching |
| `beta` | 1.0 | 1.0 below ~4x speedup, 0.5 above |
| `jf_max` | 8.0 | max jump length |

Tested on ComfyUI commit `3bfe4e52`. Not bit-exact with ComfyUI's Euler sampler at `tau 0`.
