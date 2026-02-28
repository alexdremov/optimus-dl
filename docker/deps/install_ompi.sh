#!/usr/bin/env bash
set -ex

mkdir -p /opt/openmpi
mkdir -p /setup/openmpi
cd /setup/openmpi
wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz
tar xf openmpi-5.0.10.tar.gz
cd openmpi-5.0.10
mkdir build

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

./configure \
    --with-cuda=/usr/local/cuda \
    --prefix=/opt/openmpi \
    --with-ofi=/opt/libfabric \
    --with-ucx=/opt/ucx \
    | tee config.out
make -j$(nproc) all | tee make.out
sudo make install | tee install.out
