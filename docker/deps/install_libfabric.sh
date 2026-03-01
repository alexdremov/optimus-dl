#!/usr/bin/env bash
set -ex

export libfabric_version=1.22.0

mkdir -p /opt/libfabric
git clone --branch v${libfabric_version} --depth 1 https://github.com/ofiwg/libfabric.git
cd libfabric
./autogen.sh
./configure --prefix=/opt/libfabric \
    --with-cuda=/usr/local/cuda \
    --enable-cuda-dlopen \
    --enable-gdrcopy-dlopen \
    --enable-efa
make -j$(nproc)
make install
ldconfig
cd ..
rm -rf libfabric
