#!/bin/bash

# Default version if not provided
export VERSION=${VERSION:-"dev"}

# Locally we always want to build amd64
export ARCH="amd64"
export HOST_ARCH=$(uname -m)

echo "Building for Architecture: ${ARCH} (Host: ${HOST_ARCH}) with Version: ${VERSION}"

# Navigate to the project root
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Determine ccache source override
CCACHE_DIR="$HOME/.ccache"
CCACHE_OVERRIDE=""
if [ -d "$CCACHE_DIR" ]; then
    echo "Found local ccache at $CCACHE_DIR, using it for build context."
    CCACHE_OVERRIDE="--set *.contexts.ccache_src=$CCACHE_DIR"
else
    echo "Local ccache directory not found at $CCACHE_DIR. Creating it for persistence..."
    mkdir -p "$CCACHE_DIR"
    CCACHE_OVERRIDE="--set *.contexts.ccache_src=$CCACHE_DIR"
fi

# Function to write Docker's internal cache back to your local ~/.ccache
# This runs automatically on success OR failure via the trap below.
export_ccache() {
    echo "Syncing Docker cache back to $CCACHE_DIR..."
    docker buildx bake -f docker/docker-bake.hcl ccache-export \
        --allow fs.read=$HOME \
        --allow fs.write=$HOME \
        --set "*.args.ARCH=${ARCH}" \
        --set "*.args.CACHE_BUSTER=$(date +%s)" \
        --set "ccache-export.output=type=local,dest=$CCACHE_DIR"
}

# Ensure your local ~/.ccache is updated even if the compilation fails
trap export_ccache EXIT

# Allow overriding push vs load (default to load for local builds unless specified)
ACTION="--load"
if [[ "$*" == *"--push"* ]]; then
    ACTION="--push"
fi

# Step 1: Seed the internal Docker cache mount from your local ~/.ccache
# This runs sequentially to avoid invalidating the main builder's layers.
echo "Seeding internal Docker cache mount..."
docker buildx bake -f docker/docker-bake.hcl ccache-seed \
    --allow fs.read=$HOME \
    --set "*.args.ARCH=${ARCH}" \
    --set "*.args.CACHE_BUSTER=$(date +%s)" \
    ${CCACHE_OVERRIDE}

# Step 2: Run the main build
# It uses the same cache id, so it picks up the data from Step 1.
docker buildx bake -f docker/docker-bake.hcl \
    --progress plain \
    --push \
    --allow network.host \
    --allow fs.read=$HOME \
    --allow fs.write=/tmp \
    ${ACTION} \
    --set "*.args.ARCH=${ARCH}" \
    "$@"

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "Successfully built optimus-dl targets."
else
    echo "Docker buildx bake failed with exit code $BUILD_EXIT_CODE."
fi

exit $BUILD_EXIT_CODE
