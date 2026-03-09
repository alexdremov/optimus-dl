#!/bin/bash

set -ex

# 1. Install prerequisites
apt-get update && apt-get install -y wget software-properties-common

# 2. Add the NVIDIA CUDA network repository (Example for Ubuntu 22.04)
# (If you are on Ubuntu 20.04, change 'ubuntu2204' to 'ubuntu2004' below)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update

# 3. Install DCGM
apt-get install -y datacenter-gpu-manager
