#!/bin/bash

set -ex

# 1. Detect architecture and set the NVIDIA repo path
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    NV_ARCH="x86_64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # NVIDIA uses 'sbsa' (Server Base System Architecture) for datacenter ARM
    NV_ARCH="sbsa"
else
    echo "Error: Unsupported architecture $ARCH"
    exit 1
fi

# 2. Install prerequisites
apt-get update && apt-get install -y --no-install-recommends wget software-properties-common

# 3. Add the NVIDIA CUDA network repository for the detected architecture
# (Example is for Ubuntu 22.04. Change 'ubuntu2204' to 'ubuntu2004' if needed)
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NV_ARCH}/cuda-keyring_1.0-1_all.deb"
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update

# 4. Install DCGM
apt-get install -y --no-install-recommends datacenter-gpu-manager

rm -rf /var/lib/apt/lists/*
