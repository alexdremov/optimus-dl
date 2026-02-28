#!/bin/bash

# Default version if not provided
export VERSION="dev"

echo "Building optimus-dl:${VERSION} and optimus-dl:latest..."

# Navigate to the docker directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

# Run docker buildx bake
docker buildx bake -f docker/docker-bake.hcl --progress plain --push

if [ $? -eq 0 ]; then
    echo "Successfully built and pushed optimus-dl:${VERSION} and optimus-dl:latest."
else
    echo "Docker buildx bake failed."
    exit 1
fi
