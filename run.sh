#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

IMAGE="${DIY_JEV_IMAGE:-diy-jev:cuda}"
PORT="${DIY_JEV_PORT:-8080}"
MODELS_DIR="${DIY_JEV_MODELS_DIR:-$ROOT_DIR/models}"

die() {
    echo "error: $*" >&2
    exit 1
}

# Docker must be installed.
command -v docker >/dev/null 2>&1 \
    || die "Docker is not installed."

docker info >/dev/null 2>&1 \
    || die "Docker daemon is not running or your user cannot access it."

# Persistent model storage.
mkdir -p "$MODELS_DIR"

echo "Checking NVIDIA Docker support..."

docker run --rm --gpus all \
    nvidia/cuda:12.0.1-base-ubuntu22.04 \
    nvidia-smi >/dev/null \
    || die "Docker cannot access the NVIDIA GPU. Install/configure NVIDIA Container Toolkit."

echo "Building DIY-Jev..."
docker build \
    -f Dockerfile.cuda \
    -t "$IMAGE" \
    .

echo
echo "Starting DIY-Jev..."
echo "Models: $MODELS_DIR"
echo "Server: http://127.0.0.1:$PORT"
echo

exec docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e JEV_BIND_ADDR=0.0.0.0:8080 \
    -p "$PORT:8080" \
    -v "$MODELS_DIR:/app/models" \
    "$IMAGE" \
    "$@"
