# ML Homelab Quick Start Guide

Get up and running with Ray 2.49.2 and comprehensive metrics tracking in minutes!

## Prerequisites

- Docker and Docker Compose
- Python 3.10 or 3.11
- [uv](https://github.com/astral-sh/uv) (Python package installer)

## 1. Install Python Dependencies

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

## 2. Start All Services

```bash
make start
```

That's it! This single command will:
- ✅ Start MinIO (S3-compatible storage)
- ✅ Start Prometheus (metrics collection)
- ✅ Start Grafana (metrics visualization)
- ✅ Start Ray 2.49.2 with metrics enabled
- ✅ **Automatically import Ray's Grafana dashboards**
- ✅ Start Streamlit (web UI)

## 3. Access Your Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Ray Dashboard** | http://localhost:8265/ | Monitor Ray cluster, jobs, and tasks |
| **Grafana** | http://localhost:3000/ | View metrics dashboards (admin/admin) |
| **Prometheus** | http://localhost:9090/ | Query raw metrics data |
| **Streamlit** | http://localhost:8501/ | ML job submission UI |
| **MinIO** | http://localhost:9001/ | S3 storage console |

## 4. Explore Ray Dashboards in Grafana

1. Open http://localhost:3000/ 
2. Login with `admin` / `admin`
3. Navigate to **Dashboards → Browse → Ray**
4. Explore pre-built dashboards showing:
   - CPU, memory, and disk usage
   - Task and actor metrics
   - Object store utilization
   - Job execution metrics

## 5. Run Your First ML Job

```bash
# Simple Ray job
python hello_ray_job.py

# MNIST training example
cd streamlit_app/jobs/mnist_training
./run.sh
```

Watch the metrics update in real-time in Grafana!

## Stop All Services

```bash
make stop
```

## What's Automatic?

### 🎯 Automatic Metrics Setup

When you run `make start`, the system automatically:

1. **Starts all infrastructure services**
   - MinIO for S3 storage
   - Prometheus for metrics collection
   - Grafana for visualization

2. **Starts Ray with metrics enabled**
   - Exports metrics on port 8080
   - Configures Prometheus scraping

3. **Imports Ray's default Grafana dashboards**
   - Waits for Ray to generate dashboard files
   - Copies them to Grafana's provisioning directory
   - Restarts Grafana to load the dashboards
   - **No manual intervention needed!**

4. **Starts your web interface**
   - Streamlit dashboard for job management

### 📊 Available Metrics

Ray automatically exports comprehensive metrics:

**System Metrics:**
- CPU utilization per node
- Memory usage and availability
- Disk usage
- Network I/O

**Application Metrics:**
- Task execution times and states
- Actor lifecycle and performance
- Object store usage

**Job Metrics:**
- Job submission and execution
- Resource utilization per job

### 🔍 Example Prometheus Queries

Try these in Prometheus (http://localhost:9090/):

```promql
# CPU usage
ray_node_cpu_utilization

# Memory usage percentage
100 * (ray_node_mem_used / (ray_node_mem_used + ray_node_mem_available))

# Running tasks
ray_tasks{State="RUNNING"}

# Object store usage
ray_object_store_memory
```

## Troubleshooting

### Dashboards Not Showing?

If Ray dashboards don't appear in Grafana:

```bash
# Manually re-import dashboards
make setup-metrics
```

### Services Not Starting?

Check that ports are available:
```bash
# Check if ports are in use
lsof -i :8265  # Ray
lsof -i :9090  # Prometheus
lsof -i :3000  # Grafana
```

Stop conflicting services or modify ports in `.env` file.

### Check Service Status

```bash
# Docker services
docker compose ps

# Ray status
ray status

# Check metrics endpoint
curl http://localhost:8080/metrics
```

## Next Steps

- **[Full README](README.md)** - Complete documentation
- **[Metrics Setup Guide](docs/metrics-setup.md)** - Detailed metrics configuration
- **[Docker Setup](docs/docker-setup.md)** - Docker and Docker Compose guide
- **[KubeRay Setup](docs/kuberay-setup.md)** - Production Kubernetes setup

## Architecture at a Glance

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│ Ray 2.49.2  │────▶│   ML Jobs   │
│   :8501     │     │   :8265     │     │  (Training/ │
└─────────────┘     └──────┬──────┘     │  Inference) │
                           │            └─────────────┘
                           │ metrics:8080
                    ┌──────┼──────┐
                    │      │      │
                    ▼      ▼      ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │  MinIO   │ │Prometheus│ │ Grafana  │
             │  :9000   │ │  :9090   │ │  :3000   │
             └──────────┘ └──────────┘ └──────────┘
```

## Useful Commands

```bash
# Start all services
make start

# Stop all services
make stop

# Run Streamlit only
make run

# Re-import Ray dashboards
make setup-metrics

# View logs
docker compose logs -f

# Check Ray status
ray status

# List Ray jobs
ray job list
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
METRICS_EXPORT_PORT=8080
DASHBOARD_PORT=8265
```

---

**That's it!** 🎉 You now have a fully functional ML development environment with comprehensive metrics tracking.

Start building and monitoring your distributed ML workloads! 🚀

