#!/bin/sh
# Submit the MNIST Tune job to a running Ray cluster.
#
# Mirrors the "MNIST Tune" JobSpec in streamlit_app/job_runner.py -- same
# address, working directory, entrypoint, and runtime env -- so submitting from
# the CLI and from the Streamlit dashboard runs an identical job.
#
# Requires a running cluster (`make start`) and the ray CLI (`uv run ray ...`).
set -eu

# The jobs root (streamlit_app/jobs) is one level above this script's own
# directory, resolved to an absolute path so the command works from any cwd.
JOBS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# The Jobs API speaks HTTP on the dashboard port. This is deliberately not
# ray://10001 (the Ray Client protocol), which needs the client and cluster
# Python versions to match.
ray job submit \
  --address http://127.0.0.1:8265 \
  --working-dir "$JOBS_DIR" \
  --runtime-env "$JOBS_DIR/mnist_training/runtime_env.yaml" \
  -- python mnist_training/train_mnist.py
