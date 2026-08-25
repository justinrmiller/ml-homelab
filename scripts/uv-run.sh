#!/usr/bin/env sh
# `uv run` with the PyTorch extra this machine needs.
#
# torch lives in a mutually exclusive extra, so a bare `uv run` would sync the
# environment without it. Everything that needs the project environment should
# go through here (or pass --extra itself).
set -e
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --extra "$("$script_dir/torch-extra.sh")" "$@"
