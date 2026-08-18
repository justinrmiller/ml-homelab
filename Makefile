.PHONY: run start stop status sync job test cov lint format typecheck check hooks dashboards

# Sync Python dependencies
sync:
	uv sync

# Start Streamlit app only
run:
	uv run streamlit run streamlit_app/app.py

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
	uv run pytest

# Run the test suite with coverage (fails below the floor in pyproject.toml)
cov:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

# Lint with ruff
lint:
	uv run ruff check .

# Format with ruff
format:
	uv run ruff check --fix .
	uv run ruff format .

# Type check with ty
typecheck:
	uv run ty check

# Everything CI runs
check: lint typecheck cov

# Re-export Ray's Grafana dashboards (run after bumping the Ray version)
dashboards:
	uv run python scripts/export_grafana_dashboards.py

# Install the git hooks
hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

# Submit a Ray job
# Usage:
#   make job SCRIPT=examples/hello_ray_job.py
#   make job SCRIPT=resnet_inference/inference.py RUNTIME_ENV=resnet_inference/runtime_env.yaml
RUNTIME_ENV ?=
_RUNTIME_ENV_FLAG = $(if $(RUNTIME_ENV),--runtime-env $(RUNTIME_ENV),)
job:
	uv run ray job submit --address http://localhost:8265 --working-dir . $(_RUNTIME_ENV_FLAG) -- python $(SCRIPT)
