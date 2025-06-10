#/bin/sh

# DEPRECATED, SEE README
echo "Check the README! This script is deprecated"
exit 1

# make sure to install ray prior to running this script

#ray://0.0.0.0:8265

uv run ray job submit \
  --address http://0.0.0.0:8265 \
  --working-dir . \
  --runtime-env-json '{
    "pip": {
      "packages": ["torch", "torchvision", "ray[tune]", "filelock"],
      "pip_check": false
    }
  }' \
  -- python train_mnist.py
