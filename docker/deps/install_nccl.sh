#!/usr/bin/env bash
set -ex

mkdir -p /opt
cd /opt
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.29.3-1

# Create a local bin directory for ccache symlinks to avoid space-separated NVCC variables
# which break Makefile logic that uses $(firstword $(NVCC))
mkdir -p /tmp/ccache-bin
ln -sf /usr/bin/ccache /tmp/ccache-bin/nvcc
ln -sf /usr/bin/ccache /tmp/ccache-bin/g++
ln -sf /usr/bin/ccache /tmp/ccache-bin/gcc
ln -sf /usr/bin/ccache /tmp/ccache-bin/c++
ln -sf /usr/bin/ccache /tmp/ccache-bin/cc
export PATH="/tmp/ccache-bin:$PATH"

make -j8 src.build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120" \
    CUDA_HOME="/usr/local/cuda"
make -j8 pkg.debian.build

[ -d "/opt/nccl/build" ] && [ -d "/opt/nccl/build/include" ] && [ -d "/opt/nccl/build/lib" ]
