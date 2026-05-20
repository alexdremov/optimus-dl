#!/bin/bash

docker run \
    --rm \
    -it \
    -v uv-linux-cache:/root/.cache/uv \
    -v "$(pwd)/optimus_dl:/app/optimus_dl" \
    -v "$(pwd)/pyproject.toml:/app/pyproject.toml" \
    -v "$(pwd)/uv.lock:/app/uv.lock" \
    --platform linux/amd64 \
    -w /app \
    -e GROUPED_GEMM_SKIP_CUDA_BUILD=TRUE \
    -e UV_CACHE_DIR=/root/.cache/uv \
    -e SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 \
    ghcr.io/astral-sh/uv:python3.11-bookworm-slim \
    bash -c 'apt-get update -qq && apt-get install -yq git >/dev/null 2>&1 && uv "$@"' -- "$@"
