#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

"$BUILD_DIR"/cuda-raytracer

echo "Wrote output.ppm"
