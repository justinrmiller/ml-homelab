.PHONY: run start stop status sync job test cov lint format typecheck check hooks dashboards

# torch ships as two mutually exclusive extras (cpu/gpu); this picks the one
# matching the machine. Override with `make test TORCH_EXTRA=gpu`.
TORCH_EXTRA ?= $(shell scripts/torch-extra.sh)
UV_RUN = scripts/uv-run.sh

# Sync Python dependencies
sync:
	uv sync --extra $(TORCH_EXTRA)

# Start Streamlit app only
run:
	$(UV_RUN) streamlit run streamlit_app/app.py

# Start KubeRay cluster with all services
start:
	scripts/kuberay-init.sh

# Stop KubeRay cluster and all services
stop:
	scripts/kuberay-stop.sh

# Check cluster status
status:
	scripts/kuberay-status.sh

# Run the test suite
test:
	$(UV_RUN) pytest

# Run the test suite with coverage (fails below the floor in pyproject.toml)
cov:
	$(UV_RUN) pytest --cov --cov-report=term-missing --cov-report=html

# Lint with ruff
lint:
	$(UV_RUN) ruff check .

# Format with ruff
format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

# Type check with ty
typecheck:
	$(UV_RUN) ty check

# Everything CI runs
check: lint typecheck cov

# Re-export Ray's Grafana dashboards (run after bumping the Ray version)
dashboards:
	$(UV_RUN) python scripts/export_grafana_dashboards.py

# Install the git hooks
hooks:
	$(UV_RUN) pre-commit install
	$(UV_RUN) pre-commit install --hook-type pre-push

# Submit a Ray job
# Usage:
#   make job SCRIPT=examples/hello_ray_job.py
#   make job SCRIPT=resnet_inference/inference.py RUNTIME_ENV=resnet_inference/runtime_env.yaml
RUNTIME_ENV ?=
_RUNTIME_ENV_FLAG = $(if $(RUNTIME_ENV),--runtime-env $(RUNTIME_ENV),)
job:
	$(UV_RUN) ray job submit --address http://localhost:8265 --working-dir . $(_RUNTIME_ENV_FLAG) -- python $(SCRIPT)
