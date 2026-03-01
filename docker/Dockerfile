# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Cache seeding context
# ---------------------------------------------------------------------------
FROM scratch AS ccache_src

# ---------------------------------------------------------------------------
# Builder Stage
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.02-py3 AS base

LABEL org.opencontainers.image.version=${VERSION}
ARG ARCH=amd64

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

# Install uv
ADD https://astral.sh/uv/install.sh uv-installer.sh
RUN sh ./uv-installer.sh && rm uv-installer.sh

COPY ./pyproject.toml /setup/optimus_dl/
COPY ./optimus_dl /setup/optimus_dl/optimus_dl
RUN --mount=type=cache,target=/root/.cache/uv cd /setup/optimus_dl/ && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    pip install .[dev] \
        --no-cache \
        --no-build-isolation \
    && chmod -R 777 /usr/local/lib/python3.12/dist-packages \
    && chmod -R 777 /app/venv

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_DEFAULT_TIMEOUT=100

RUN echo TF_FORCE_GPU_ALLOW_GROWTH='true' >> /etc/environment

# Root password
RUN echo 'root:root' | chpasswd

WORKDIR /app

# Entrypoint
ENV OMP_NUM_THREADS=1
CMD ["/bin/zsh"]

FROM base AS interactive

WORKDIR /setup
COPY docker/requirements-dev.txt /setup/requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /setup/requirements-dev.txt --no-cache

RUN curl https://getmic.ro | bash
RUN mv ./micro /usr/local/bin/micro

RUN --mount=target=/var/lib/apt/lists,type=cache \
    --mount=target=/var/cache/apt,type=cache <<EOR
    set -ex
    rm -f /etc/apt/apt.conf.d/docker-clean
    apt update
    apt install -y software-properties-common curl wget ca-certificates locales sudo git openssh-client rsync unzip tmux htop nvtop libsm6 libxext6 psmisc
    apt-add-repository ppa:fish-shell/release-3
    apt update
    apt install -y fish

    # Clean up
    rm -rf /var/lib/apt/lists/*
    apt autoremove && apt clean
EOR

RUN grep -q '^/bin/fish$' /etc/shells; or echo '/bin/fish' | tee -a /etc/shells
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

# ---------------------------------------------------------------------------
# Ccache Storage Image Stage
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS ccache-export
ARG ARCH=amd64
ARG CACHE_BUSTER=0
RUN --mount=type=cache,target=/ccache,id=ccache-${ARCH} \
    echo "Exporting ccache (buster: ${CACHE_BUSTER})..." && \
    mkdir -p /export && cp -a /ccache/. /export/ || true

FROM busybox AS ccache-only
COPY --from=ccache-export /export /root/.ccache

# ---------------------------------------------------------------------------
# Ccache Seeding Stage (Used to warm the cache mount sequentially)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS ccache-seed
RUN apt-get update && apt-get install -y rsync && rm -rf /var/lib/apt/lists/*
ARG ARCH=amd64
ARG CACHE_BUSTER=0
# We use FROM scratch AS ccache_src as a placeholder
RUN --mount=type=cache,target=/ccache,id=ccache-${ARCH} \
    --mount=type=bind,from=ccache_src,target=/src \
    echo "Seeding ccache (buster: ${CACHE_BUSTER})..." && \
    if [ -d /src/.ccache ]; then \
        echo "Found registry ccache, seeding internal mount..."; \
        rsync -a /src/.ccache/ /ccache/; \
    elif [ -d /src ]; then \
        echo "Found source directory, attempting to seed from it directly..."; \
        rsync -a /src/ /ccache/; \
    else \
        echo "No registry ccache found to seed."; \
    fi
