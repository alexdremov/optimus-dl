#!/usr/bin/env bash
set -ex

ARCH=$(uname -m)
mkdir -p /opt

if [ "$ARCH" = "x86_64" ]; then
    wget -q https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
    tar xpzf nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz
    mv nvhpc_2026_261_Linux_x86_64_cuda_multi /opt/nvhpc_install
elif [ "$ARCH" = "aarch64" ]; then
    wget -q https://developer.download.nvidia.com/hpc-sdk/26.1/nvhpc_2026_261_Linux_aarch64_cuda_multi.tar.gz
    tar xpzf nvhpc_2026_261_Linux_aarch64_cuda_multi.tar.gz
    mv nvhpc_2026_261_Linux_aarch64_cuda_multi /opt/nvhpc_install
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

ls /opt/nvhpc_install
