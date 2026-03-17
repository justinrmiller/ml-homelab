# ML Homelab Quick Start Guide

Get up and running with KubeRay, MinIO, and metrics monitoring in minutes!

## Prerequisites

- Docker or Podman
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

The setup script will automatically install Kind, Helm, and kubectl via Homebrew if missing.

## 1. Install Python Dependencies

```bash
# Sync dependencies (uv creates the venv automatically)
uv sync
```

> **Note:** `uv sync` automatically creates a `.venv` virtual environment and installs all dependencies from `pyproject.toml` and `uv.lock`. No need to manually create or activate a venv — all `make` targets use `uv run` which handles this transparently.

## 2. Start All Services

```bash
make start
```

This single command will:
- Check prerequisites and sync Python dependencies via `uv`
- Create a local Kubernetes cluster (Kind)
- Install KubeRay operator and Ray cluster
- Start MinIO (S3-compatible storage)
- Start Prometheus (metrics collection)
- Start Grafana (metrics visualization)
- Set up port forwarding for Ray services
- Start Streamlit (web UI)

## 3. Access Your Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Ray Dashboard** | http://localhost:8265/ | Monitor Ray cluster, jobs, and tasks |
| **Grafana** | http://localhost:3000/ | View metrics dashboards (admin/admin) |
| **Prometheus** | http://localhost:9090/ | Query raw metrics data |
| **Streamlit** | http://localhost:8501/ | ML job submission UI |
| **MinIO** | http://localhost:9001/ | S3 storage console |

## 4. Run Your First ML Job

```bash
# Submit a simple Ray job
uv run ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"

# Or use the Makefile shortcut
make job SCRIPT=hello_ray_job.py

# MNIST training example
cd streamlit_app/jobs/mnist_training
./run.sh
```

## 5. Check Cluster Status

```bash
make status
```

## Stop All Services

```bash
make stop
```

## Troubleshooting

### Services Not Starting?

Check that Docker/Podman is running and ports are available:
```bash
lsof -i :8265  # Ray
lsof -i :9090  # Prometheus
lsof -i :3000  # Grafana
lsof -i :9000  # MinIO
```

### Check Detailed Status

```bash
# Full cluster status
make status

# Kubernetes pods
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay

# Ray head logs
kubectl logs -l ray.io/node-type=head

# Container service logs (Docker or Podman)
docker compose logs -f       # Docker
podman-compose logs -f       # Podman
```

## Useful Commands

```bash
make sync     # Sync Python dependencies with uv
make start    # Start all services
make stop     # Stop all services
make status   # Check cluster status
make run      # Run Streamlit only
make job SCRIPT=hello_ray_job.py  # Submit a Ray job
```

## Environment Configuration

Create a `.env` file to customize ports and credentials:

```env
# MinIO
MINIO_ROOT_USER=test-key
MINIO_ROOT_PASSWORD=test-secret

# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Ray
DASHBOARD_PORT=8265
HEAD_NODE_PORT=10001
PORT=6379
RAY_ADDRESS=ray://127.0.0.1:10001
```

## Next Steps

- **[Full README](README.md)** - Complete documentation
- **[KubeRay Setup](docs/kuberay-setup.md)** - Detailed KubeRay configuration
