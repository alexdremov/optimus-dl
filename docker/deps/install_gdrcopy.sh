#!/usr/bin/env bash
set -ex

export gdrcopy_version=2.5.1
export CUDA_PATH=/usr/local/cuda

git clone --depth 1 --branch v${gdrcopy_version} https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make CC=gcc CUDA=$CUDA_PATH lib
make lib_install
cd ../
rm -rf gdrcopy

