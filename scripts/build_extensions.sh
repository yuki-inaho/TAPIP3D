#!/usr/bin/env bash
# Build the local CUDA extensions into the active uv venv.
#
# IMPORTANT: re-run this after every `uv sync`. uv removes any package that is
# not tracked in uv.lock, and pointops2 is built from third_party/ (not PyPI),
# so each `uv sync` uninstalls it.
#
# Override the target GPU arch via TORCH_CUDA_ARCH_LIST (default 6.1 = GTX 10xx;
# 7.5 = RTX 20xx, 8.6 = RTX 30xx, 8.9 = RTX 40xx).
set -euo pipefail

ARCH="${TORCH_CUDA_ARCH_LIST:-6.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building pointops2 (TORCH_CUDA_ARCH_LIST=$ARCH, CUDA_HOME=$CUDA_HOME)..."
(
  cd "$ROOT/third_party/pointops2"
  TORCH_CUDA_ARCH_LIST="$ARCH" CUDA_HOME="$CUDA_HOME" uv run python setup.py install
)

echo "pointops2 installed."
echo "(Monocular / MegaSAM users: also build third_party/megasam/base the same way.)"
