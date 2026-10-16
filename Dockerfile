# docker build -t softcache .
# docker run --rm --gpus all -e HF_TOKEN -v ~/.cache/huggingface:/hf -v "$PWD":/work -w /work \
#     softcache --prompts prompts.json --out outputs      # needs the NVIDIA Container Toolkit

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=3.11 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH HF_HOME=/hf

WORKDIR /app
COPY . .
# third_party/SeaCache is a git submodule: clone with --recursive before building
RUN test -e third_party/SeaCache/FLUX && uv sync --frozen --no-dev

ENTRYPOINT ["softcache"]
