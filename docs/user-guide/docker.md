# Docker & Deployment

Optimus-DL provides a comprehensive set of Docker integration scripts and Dockerfiles designed for high-performance distributed training. This guide explains how to build, run, and deploy your training environment using Docker.

## 🏗 Build Your Environment

The `docker/` directory contains everything needed to build a containerized training environment.

- `docker/Dockerfile`: The main Dockerfile, which can be customized.
- `docker/build.sh`: A helper script for building Docker images.
- `docker/docker-bake.hcl`: A configuration file for Docker Buildx Bake, used for building multi-platform or multi-target images.

### Quick Build

To build the default training image:

```bash
bash docker/build.sh
```

This will create an image named `optimus-dl:latest` (by default) with all the necessary dependencies installed, including FlashAttention, NCCL, and GPU-optimized libraries.

## 🚀 Running Your Container

When running the container, ensure you provide access to your GPUs and mount your project directory.

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -w /workspace \
  optimus-dl:latest /bin/bash
```

Inside the container, you can run training jobs just like in your local environment.
