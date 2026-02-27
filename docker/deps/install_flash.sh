#!/usr/bin/env bash
set -ex

git clone https://github.com/HazyResearch/flash-attention
cd flash-attention
git checkout $FLASH_COMMIT
uv pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation ./
