#!/usr/bin/env bash
set -ex

git clone https://github.com/HazyResearch/flash-attention
cd flash-attention
git checkout $FLASH_COMMIT

# Create a local bin directory for ccache symlinks to avoid space-separated NVCC variables
# which break Makefile logic that uses $(firstword $(NVCC))
rm -rf /tmp/ccache-bin || true
mkdir -p /tmp/ccache-bin
ln -sf /usr/bin/ccache /tmp/ccache-bin/nvcc
ln -sf /usr/bin/ccache /tmp/ccache-bin/g++
ln -sf /usr/bin/ccache /tmp/ccache-bin/gcc
ln -sf /usr/bin/ccache /tmp/ccache-bin/c++
ln -sf /usr/bin/ccache /tmp/ccache-bin/cc
export PATH="/tmp/ccache-bin:$PATH"


export FLASH_ATTN_CUDA_ARCHS="80;90;100;110"
export MAX_JOBS=1
export CUDA_HOME=/usr/local/cuda
uv run python setup.py install
