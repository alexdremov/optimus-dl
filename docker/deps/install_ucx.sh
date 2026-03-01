#!/usr/bin/env bash
set -ex

# Install UCX
export UCX_VERSION=1.19.0
wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz \
    && tar xzf ucx-${UCX_VERSION}.tar.gz \
    && cd ucx-${UCX_VERSION} \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/opt/ucx --with-cuda=/usr/local/cuda --with-gdrcopy=/usr/local \
       --enable-mt --enable-devel-headers \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf ucx-${UCX_VERSION}.tar.gz ucx-${UCX_VERSION}
