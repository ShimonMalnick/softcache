"""Generate FLUX.1-dev images, or HunyuanVideo / Wan2.1 videos, with SoftCache for every
prompt in a JSON file."""
import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxPipeline, HunyuanVideoPipeline, WanPipeline
from diffusers.utils import export_to_video

from . import flux, hunyuan, wan
from .method import Config

# family -> (pipeline class, sampler module, default model, default size + guidance, video fps)
FAMILIES = {
    "flux": (FluxPipeline, flux, "black-forest-labs/FLUX.1-dev",
             dict(height=1024, width=1024, guidance=3.5), None),
    "hunyuan": (HunyuanVideoPipeline, hunyuan, "hunyuanvideo-community/HunyuanVideo",
                dict(height=480, width=640, num_frames=65, guidance=6.0), 8),
    "wan": (WanPipeline, wan, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            dict(height=480, width=832, num_frames=81, guidance=6.0), 16),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-family", choices=FAMILIES, default="flux")
    ap.add_argument("--prompts", default="prompts.json",
                    help='JSON list of {"prompt": str, "seed": int}')
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", help="model id or local snapshot directory")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--num-frames", type=int, help="video only")
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--tau", type=float, default=0.30,
                    help="staleness threshold; higher is faster. 0 = no caching (reference)")
    ap.add_argument("--jf-max", type=float, default=8.0, help="cap on the jump factor")
    ap.add_argument("--beta", type=float, default=1.0, help="Residual Motion step")
    ap.add_argument("--device", default="cuda:0", help="GPU to run on")
    ap.add_argument("--low-memory", action="store_true",
                    help="encode all prompts first, then move the text encoders to the CPU")
    args = ap.parse_args()

    pipeline_cls, sampler, default_model, defaults, fps = FAMILIES[args.model_family]
    options = {k: getattr(args, k) if getattr(args, k) is not None else v
               for k, v in defaults.items()}
    items = json.loads(Path(args.prompts).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(tau=args.tau, jf_max=args.jf_max, beta=args.beta)
    pipe = pipeline_cls.from_pretrained(args.model or default_model, torch_dtype=torch.bfloat16)

    prompts = [item["prompt"] for item in items]
    if args.low_memory:
        # Encode every prompt first, then move the text encoders off the GPU, so the transformer
        # fits next to the VAE.
        encoders = [m for m in (pipe.text_encoder, getattr(pipe, "text_encoder_2", None))
                    if m is not None]
        for m in encoders:
            m.to(args.device)
        prompts = [sampler.encode(pipe, p) for p in prompts]
        for m in encoders:
            m.to("cpu")
        torch.cuda.empty_cache()
        pipe.transformer.to(args.device)
        pipe.vae.to(args.device)
    else:
        pipe.to(args.device)
    if fps:
        pipe.vae.enable_tiling()

    for k, (item, prompt) in enumerate(zip(items, prompts)):
        seed = item.get("seed", 0)
        sample, stats = sampler.generate(pipe, prompt, seed, cfg, args.steps, **options)
        name = f"{k:03d}_seed{seed}"
        if fps:
            export_to_video(sample, str(out / f"{name}.mp4"), fps=fps)
        else:
            sample.save(out / f"{name}.png")
        stats = {"prompt": item["prompt"], "seed": seed, **stats}
        (out / f"{name}.json").write_text(json.dumps(stats, indent=1))
        print(f"{name}: actions {stats['actions']}")


if __name__ == "__main__":
    main()
