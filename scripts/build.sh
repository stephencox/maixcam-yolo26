#!/usr/bin/env bash
# Build or release with MaixCDK v4.12.5 in sipeed/maixcdk-builder.
# Usage: scripts/build.sh build|release
set -euo pipefail
MODE=${1:-build}
PIN=db6b17821b900c97eaa899be548ddc1cba0dd635
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SDK=${MAIXCDK_CACHE:-$HOME/.cache/maixcam-yolo26/MaixCDK}
if [ ! -d "$SDK/.git" ]; then
    mkdir -p "$(dirname "$SDK")"
    git clone --filter=blob:none https://github.com/sipeed/MaixCDK "$SDK"
fi
if [ "$(git -C "$SDK" rev-parse HEAD)" != "$PIN" ]; then
    git -C "$SDK" fetch origin "$PIN" && git -C "$SDK" checkout --detach "$PIN"
fi
[ -z "$(git -C "$SDK" status --porcelain)" ] || { echo "SDK cache $SDK is dirty; refusing" >&2; exit 1; }
case "$MODE" in
    build) CMD="maixcdk build -p maixcam" ;;
    release) CMD="maixcdk release -p maixcam" ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
docker run --rm --network=host -e MAIXCDK_PATH=/opt/MaixCDK \
    -v "$SDK":/opt/MaixCDK -v "$REPO":/work/app \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --entrypoint bash sipeed/maixcdk-builder -c "cd /work/app && $CMD"
