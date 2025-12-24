FROM nvidia/cuda:12.6.3-base-ubuntu24.04 AS base

ARG VERSION=latest
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

    apt update
    apt install -y software-properties-common curl wget ca-certificates locales sudo git openssh-client rsync unzip tmux htop nvtop
    apt-add-repository ppa:fish-shell/release-3
    add-apt-repository ppa:deadsnakes/ppa
    apt update

    # Install Python 3.12 and shells (zsh/fish needed for entrypoint)
    apt install -y python3-pip python3.12 python3.12-venv python3.12-dev zsh fish build-essential $(cat packages.txt)

    # Clean up
    rm -rf /var/lib/apt/lists/*
    apt autoremove && apt clean
EOR

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

# Create Venv
ADD https://astral.sh/uv/install.sh uv-installer.sh

# Run the installer then remove it
RUN sh ./uv-installer.sh && rm uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv -p=3.12 /app/venv
ENV PATH="/app/venv/bin:$PATH"
ARG UV_LINK_MODE=copy
ENV UV_NO_CACHE=1

COPY ./pyproject.toml /setup/optimus_dl/
COPY ./optimus_dl /setup/optimus_dl/optimus_dl
RUN cd /setup/optimus_dl/ && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv pip install .[dev] --no-cache && \
    chmod -R 777 /app/venv

WORKDIR /app
RUN rm -rf /setup

# Entrypoint
ENV OMP_NUM_THREADS=1
CMD ["/bin/zsh"]

FROM base as interactive

WORKDIR /setup
COPY docker/requirements-dev.txt /setup/requirements-dev.txt
RUN uv pip install -r /setup/requirements-dev.txt --no-cache

RUN <<EOR
    apt update
    apt install -y ffmpeg libsm6 libxext6 psmisc
EOR

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
