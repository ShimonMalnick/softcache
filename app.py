"""Gradio demo: SoftCache next to uncached FLUX.1-dev.

    uv run --with gradio python app.py [--model <id or local dir>] [--share]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import FluxPipeline
from PIL import Image

from softcache import Config
from softcache.flux import encode, generate

# The paper's three FLUX operating points (DrawBench-200, measured wall-clock speedup) -> tau.
PRESETS = {"~3.8x": 0.5, "~5.4x": 0.8, "~6.9x": 1.1}
ACTION_NAMES = "F = fresh, C = cache, J = jump"

DEFAULT_PROMPT = "A zebra to the right of a fire hydrant."
pipe = None
EMBEDS = {}   # prompt -> text embeddings, so each prompt pays the encoder weight moves once
CACHE = Path(__file__).parent / "demo_cache"   # finished renders, reused across clicks and restarts


def load(model):
    """Transformer and VAE live on the GPU; the text encoders visit it only to encode."""
    global pipe
    pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")


def encode_on_gpu(prompt):
    """The transformer steps aside while the text encoders are on the GPU, so a 32 GB card fits both."""
    if prompt in EMBEDS:
        return EMBEDS[prompt]
    pipe.transformer.to("cpu")
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    embeds = encode(pipe, prompt)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.transformer.to("cuda")
    torch.cuda.empty_cache()
    EMBEDS[prompt] = embeds
    return embeds


def timed(embeds, seed, cfg, steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image, stats = generate(pipe, embeds, seed, cfg, steps)
    torch.cuda.synchronize()
    return image, stats, time.perf_counter() - t0


def beta_for(tau):
    """The paper's rule: Residual Motion step 1.0 up to ~4x speedup (tau 0.5), 0.5 above."""
    return 1.0 if tau <= 0.5 else 0.5


def render(prompt, seed, steps, tau):
    """(image, stats incl. seconds). Saved on disk, so a repeat click, or the uncached reference
    (tau 0) of an earlier run, costs nothing."""
    key = hashlib.sha1(json.dumps([prompt, seed, steps, tau]).encode()).hexdigest()[:16]
    png, meta = CACHE / f"{key}.png", CACHE / f"{key}.json"
    if meta.exists():
        return Image.open(png), json.loads(meta.read_text())
    embeds = encode_on_gpu(prompt)
    image, stats, seconds = timed(embeds, seed, Config(tau=tau, beta=beta_for(tau)), steps)
    stats = {**stats, "seconds": seconds}
    CACHE.mkdir(exist_ok=True)
    image.save(png)
    meta.write_text(json.dumps(stats))
    return image, stats


def run(prompt, seed, tau, steps, with_reference):
    seed, steps = int(seed), int(steps)
    image, stats = render(prompt, seed, steps, tau)
    seconds = stats["seconds"]
    lines = [f"**SoftCache:** {seconds:.1f} s, {stats['full']} fresh / {stats['cached']} cached "
             f"nodes, {stats['jumps']} jumps",
             f"`{stats['actions']}`  ({ACTION_NAMES})"]
    ref_out = gr.Image(value=None, label="Uncached FLUX.1-dev")
    ours_out = gr.Image(value=image, label=f"SoftCache: {seconds:.1f} s")
    if with_reference:
        ref_image, ref = render(prompt, seed, steps, 0.0)
        speedup = ref["seconds"] / seconds
        ref_out = gr.Image(value=ref_image, label=f"Uncached FLUX.1-dev: {ref['seconds']:.1f} s")
        ours_out = gr.Image(value=image, label=f"SoftCache: {seconds:.1f} s ({speedup:.2f}x faster)")
        lines.insert(0, f"**Measured speedup on this run: {speedup:.2f}x** "
                        f"(uncached {ref['seconds']:.1f} s, SoftCache {seconds:.1f} s)")
    return ref_out, ours_out, "\n\n".join(lines)


def warmup():
    """A short throwaway run, so CUDA's one-off start-up cost is not in the first reported time."""
    timed(encode_on_gpu(DEFAULT_PROMPT), 0, Config(tau=0.5), 10)


def build_ui():
    first = "~5.4x"   # the band the zebra is shown at on the project page
    with gr.Blocks(title="SoftCache") as demo:
        gr.Markdown("# SoftCache\nTraining-free acceleration for FLUX.1-dev.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, scale=4)
            seed = gr.Number(label="Seed", value=0, precision=0, scale=1)
        preset = gr.Radio(list(PRESETS), value=first, label="Speed")
        with gr.Accordion("Advanced", open=False):
            tau = gr.Slider(0.05, 2.0, value=PRESETS[first], step=0.05,
                            label="tau (staleness threshold; higher is faster)")
            steps = gr.Slider(10, 50, value=50, step=1, label="steps")
        with_reference = gr.Checkbox(label="Also run uncached FLUX (doubles the wait)", value=False)
        go = gr.Button("Generate", variant="primary")
        with gr.Row():
            ref_out = gr.Image(label="Uncached FLUX.1-dev")
            ours_out = gr.Image(label="SoftCache")
        info = gr.Markdown()

        preset.change(PRESETS.get, preset, tau)
        go.click(run, [prompt, seed, tau, steps, with_reference],
                 [ref_out, ours_out, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--share", action="store_true", help="public gradio.live link (72 h)")
    args = ap.parse_args()
    load(args.model)
    warmup()
    build_ui().queue(max_size=8).launch(share=args.share)
