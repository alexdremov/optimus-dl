#!/usr/bin/env bash
set -ex

export NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk
/opt/nvhpc_install/install
rm -rf /opt/nvhpc_install
