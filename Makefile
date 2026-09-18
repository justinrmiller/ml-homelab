.PHONY: sync run app app-stop \
        services-up services-down services-status services-logs \
        floci-up floci-down floci-logs \
        kuberay-start kuberay-stop kuberay-status start stop status \
        ray-setup ray-head ray-worker ray-stop ray-status \
        job memray test cov lint format typecheck check hooks dashboards

# torch ships as two mutually exclusive extras (cpu/gpu); this picks the one
# matching the machine. Override with `make test TORCH_EXTRA=gpu`.
TORCH_EXTRA ?= $(shell scripts/torch-extra.sh)
UV_RUN = scripts/uv-run.sh

# Sync Python dependencies
sync:
	uv sync --extra $(TORCH_EXTRA)

# --- Services: Floci (S3), Prometheus, Grafana -----------------------------
# Independent of the Ray topology, so they get their own targets.

services-up:
	scripts/services.sh up

services-down:
	scripts/services.sh down

services-status:
	scripts/services.sh status

services-logs:
	scripts/services.sh logs

# Floci on its own, for when you only want S3
floci-up:
	scripts/services.sh up floci

floci-down:
	scripts/services.sh down floci

floci-logs:
	scripts/services.sh logs floci

# --- Streamlit dashboard ---------------------------------------------------

app:
	scripts/streamlit.sh start

app-stop:
	scripts/streamlit.sh stop

# Run Streamlit in the foreground (Ctrl-C to quit)
run:
	$(UV_RUN) streamlit run streamlit_app/app.py

# --- KubeRay topology: one machine, Ray on Kubernetes ----------------------
# Brings up the Kind cluster, KubeRay, the Compose services, and Streamlit.

kuberay-start:
	scripts/kuberay-init.sh

kuberay-stop:
	scripts/kuberay-stop.sh

kuberay-status:
	scripts/kuberay-status.sh

# Back-compat aliases for the KubeRay topology
start: kuberay-start
stop: kuberay-stop
status: kuberay-status

# --- Standalone Ray topology: many machines, no Kubernetes -----------------
# See docs/multi-machine.md. Does not touch Compose or Streamlit.

# Prepare this machine to run a Ray node (head or worker)
ray-setup:
	scripts/ray-node-setup.sh

# Start a native Ray head here, for workers on other machines to join
ray-head:
	scripts/ray-head-start.sh

# Join this machine to a head. Usage: make ray-worker HEAD=192.168.1.10
HEAD ?=
ray-worker:
	scripts/ray-worker-start.sh $(HEAD)

# Stop the Ray node running on this machine
ray-stop:
	scripts/ray-node-stop.sh

# Cluster membership and resources, from any node
ray-status:
	$(UV_RUN) ray status

# --- Development -----------------------------------------------------------

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

# Profile a synthetic workload with memray (writes a .bin capture + flamegraph)
MEMRAY_OUT ?= .memray
memray:
	$(UV_RUN) python -m examples.memray_example --output-dir $(MEMRAY_OUT)

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
# Reserving a CPU for the entrypoint is what keeps the driver off the head
# node, which is too small to host one. See ENTRYPOINT_NUM_CPUS in
# streamlit_app/job_runner.py for the whole story.
ENTRYPOINT_NUM_CPUS ?= 1
job:
	$(UV_RUN) ray job submit --address http://localhost:8265 --working-dir . \
		--entrypoint-num-cpus $(ENTRYPOINT_NUM_CPUS) $(_RUNTIME_ENV_FLAG) -- python $(SCRIPT)
