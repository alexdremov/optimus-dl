# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Source Fetching Stage (Shared across platforms)
# ---------------------------------------------------------------------------
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS source-fetcher

# Basic env for fetcher
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y git curl ca-certificates

# Install uv for cloning logic
ADD https://astral.sh/uv/install.sh /setup/uv-installer.sh
RUN sh /setup/uv-installer.sh && rm /setup/uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /setup
COPY docker/deps/clone_torch.sh /setup/clone_torch.sh
ENV PYTORCH_COMMIT=v2.10.0
RUN chmod +x /setup/clone_torch.sh && /setup/clone_torch.sh

# ---------------------------------------------------------------------------
# Builder Stage
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_DEFAULT_TIMEOUT=100
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=0
ENV CCACHE_COMPRESS=1
ENV CCACHE_COMPRESSLEVEL=6
ENV CCACHE_SLOPINESS=include_file_ctime,include_file_mtime,time_macros
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

WORKDIR /setup

# Setup basic system for python and shell
COPY docker/packages.txt packages.txt
RUN <<EOR
    set -ex
    rm -f /etc/apt/apt.conf.d/docker-clean
EOR

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && \
    apt install -y software-properties-common curl wget ca-certificates locales sudo git openssh-client rsync unzip tmux htop nvtop ccache && \
    apt-add-repository ppa:fish-shell/release-3 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3-pip python3.12 python3.12-venv python3.12-dev zsh fish build-essential $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Create Venv
ADD https://astral.sh/uv/install.sh uv-installer.sh

# Run the installer then remove it
RUN sh ./uv-installer.sh && rm uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv -p=3.12 /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
ENV VIRTUAL_ENV=/app/venv

RUN apt remove libnccl-dev libnccl2 --allow-change-held-packages -y

# Python packages
ENV NCCL_VERSION=2.29.3-1
ENV NCCL_VERSION_CODE=22903
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.29.3-1+cuda12.9
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.29.3-1+cuda12.9
COPY docker/deps/install_nccl.sh /setup/install_nccl.sh
RUN --mount=type=cache,target=/root/.ccache \
    export PATH="/usr/lib/ccache:$PATH" && \
    chmod +x /setup/install_nccl.sh && /setup/install_nccl.sh

RUN ls /opt/nccl/build/pkg/deb/libnccl2*.deb | xargs -L 1 dpkg -i
RUN ls /opt/nccl/build/pkg/deb/libnccl-dev*.deb | xargs -L 1 dpkg -i

ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0"
# Copy Pytorch source from source-fetcher (cloned once)
COPY --from=source-fetcher /setup/pytorch /setup/pytorch
COPY docker/deps/install_torch.sh /setup/install_torch.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    export PATH="/usr/lib/ccache:$PATH" && \
    chmod +x /setup/install_torch.sh && /setup/install_torch.sh

ENV FLASH_COMMIT=v2.8.3
COPY docker/deps/install_flash.sh /setup/install_flash.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    export PATH="/usr/lib/ccache:$PATH" && \
    chmod +x /setup/install_flash.sh && /setup/install_flash.sh

COPY ./pyproject.toml /setup/optimus_dl/
COPY ./optimus_dl /setup/optimus_dl/optimus_dl
RUN --mount=type=cache,target=/root/.cache/uv cd /setup/optimus_dl/ && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv pip install .[dev] --no-cache && \
    chmod -R 777 /app/venv

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS base

LABEL org.opencontainers.image.version=${VERSION}

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_DEFAULT_TIMEOUT=100

RUN echo TF_FORCE_GPU_ALLOW_GROWTH='true' >> /etc/environment

WORKDIR /setup

# Setup basic system for python and shell
COPY docker/packages.txt packages.txt
RUN <<EOR
    set -ex
    rm -f /etc/apt/apt.conf.d/docker-clean
EOR

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && \
    apt install -y software-properties-common curl wget ca-certificates locales sudo git openssh-client rsync unzip tmux htop nvtop && \
    apt-add-repository ppa:fish-shell/release-3 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3-pip python3.12 python3.12-venv python3.12-dev zsh fish build-essential $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Root password
RUN echo 'root:root' | sudo chpasswd

# Python Alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 10

# Install uv
ADD https://astral.sh/uv/install.sh uv-installer.sh

# Run the installer then remove it
RUN sh ./uv-installer.sh && rm uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY --from=builder /app/venv /app/venv

COPY --from=builder /opt/nccl /opt/nccl
RUN apt remove libnccl-dev libnccl2 --allow-change-held-packages -y || true
RUN ls /opt/nccl/build/pkg/deb/libnccl2*.deb | xargs -L 1 dpkg -i
RUN ls /opt/nccl/build/pkg/deb/libnccl-dev*.deb | xargs -L 1 dpkg -i

ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/venv

WORKDIR /app
RUN rm -rf /setup

# Entrypoint
ENV OMP_NUM_THREADS=1
CMD ["/bin/zsh"]

FROM base AS interactive

WORKDIR /setup
COPY docker/requirements-dev.txt /setup/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /setup/requirements-dev.txt --no-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && \
    apt install -y ffmpeg libsm6 libxext6 psmisc && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://getmic.ro | bash
RUN mv ./micro /usr/local/bin/micro

RUN grep -q '^/bin/fish$' /etc/shells; or echo '/bin/fish' | sudo tee -a /etc/shells
ENV SHELL=/bin/fish
SHELL ["/bin/fish", "-c"]
RUN curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher
RUN fisher install IlanCosman/tide@v6
RUN tide configure \
    --auto \
    --style=Lean \
    --prompt_colors='True color' \
    --show_time=No \
    --lean_prompt_height='Two lines' \
    --prompt_connection=Disconnected \
    --prompt_spacing=Compact \
    --icons='Few icons' \
    --transient=No

WORKDIR /app
RUN rm -rf /setup
# Entrypoint
ENV OMP_NUM_THREADS=1
CMD ["/bin/zsh"]
