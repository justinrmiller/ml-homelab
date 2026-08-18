#!/usr/bin/env sh
# Print the uv extra matching this machine's PyTorch build: `gpu` when a usable
# NVIDIA GPU is present, `cpu` otherwise.
#
# Hardware cannot be expressed as a dependency marker, so the choice has to be
# made at sync time rather than baked into uv.lock. Override with TORCH_EXTRA.
set -e

if [ -n "${TORCH_EXTRA:-}" ]; then
  printf '%s\n' "$TORCH_EXTRA"
  exit 0
fi

# CUDA wheels are Linux-only; macOS gets the CPU/MPS build from PyPI either way.
if [ "$(uname -s)" = "Linux" ] &&
   command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  printf 'gpu\n'
else
  printf 'cpu\n'
fi
