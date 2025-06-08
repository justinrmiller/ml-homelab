#/bin/sh

# make sure to install ray prior to running this script

uv run ray job submit \
  --address ray://127.0.0.1:8265 \
  --working-dir . \
  --runtime-env-json '{
    "pip": {
      "packages": ["torch", "torchvision", "ray[tune]", "filelock"],
      "pip_check": false
    }
  }' \
  -- python train_mnist.py
