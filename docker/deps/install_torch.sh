#!/usr/bin/env bash
set -ex

uv pip uninstall torch

uv pip install --no-cache-dir \
  astunparse numpy ninja pyyaml \
  setuptools cmake cffi typing_extensions \
  future six requests dataclasses

cd pytorch
export CMAKE_PREFIX_PATH="/usr/local"
export NCCL_ROOT="/opt/nccl/build"
export NCCL_INCLUDE_DIR="/opt/nccl/build/include"

export WIN32=0
export USE_CUDA=1
export USE_CUPTI_SO=1
export USE_KINETO=1
export USE_SYSTEM_NCCL=1
export USE_MPI=1
export USE_XNNPACK=0
export CFLAGS="-fno-gnu-unique -DPYTORCH_C10_DRIVER_API_SUPPORTED=1"
export MAX_JOBS=4

# build fails
sed -i '2i #include <cuda.h>' torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp

uv pip install . -v --no-build-isolation

ccache -s

cd ..
rm -rf pytorch
