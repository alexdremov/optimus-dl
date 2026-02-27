#!/usr/bin/env bash
set -ex

git clone https://github.com/pytorch/pytorch pytorch
cd pytorch
git checkout $PYTORCH_COMMIT

git submodule sync
git submodule update --init --recursive
