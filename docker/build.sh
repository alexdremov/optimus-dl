#!/bin/bash

# Default version if not provided
export VERSION=${VERSION:-"dev"}

# Detect host architecture
HOST_ARCH=$(uname -m)
case ${HOST_ARCH} in
    x86_64)  ARCH="amd64" ;;
    arm64)   ARCH="arm64" ;;
    aarch64) ARCH="arm64" ;;
    *)       ARCH="amd64" ;;
esac

echo "Building for Architecture: ${ARCH} (Host: ${HOST_ARCH}) with Version: ${VERSION}"

# Navigate to the project root
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Determine ccache source override
CCACHE_OVERRIDE=""
if [ -d "$HOME/.ccache" ]; then
    echo "Found local ccache at $HOME/.ccache, using it for build context."
    CCACHE_OVERRIDE="--set *.contexts.ccache_src=$HOME/.ccache"
fi

# Allow overriding push vs load (default to load for local builds unless specified)
ACTION="--load"
if [[ "$*" == *"--push"* ]]; then
    ACTION="--push"
fi

# Run docker buildx bake
# We explicitly set ARCH and CCACHE_SRC
docker buildx bake -f docker/docker-bake.hcl \
    --progress plain \
    ${ACTION} \
    --set "*.args.ARCH=${ARCH}" \
    ${CCACHE_OVERRIDE} \
    "$@"

if [ $? -eq 0 ]; then
    echo "Successfully built optimus-dl targets."
else
    echo "Docker buildx bake failed."
    exit 1
fi
