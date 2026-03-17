# ML Homelab

A local development environment for orchestrating, training, and visualizing machine learning workflows using KubeRay and Streamlit. This project provides a reproducible setup for running distributed ML experiments on Kubernetes, with S3-compatible storage via MinIO and metrics monitoring via Prometheus and Grafana.

---

## Project Structure

```
.
├── data/                        # Data storage for services (auto-created)
│   ├── minio/                   # MinIO S3-compatible storage
│   ├── prometheus/              # Prometheus time-series data
│   └── grafana/                 # Grafana configuration data
├── docs/                        # Documentation
│   └── kuberay-setup.md         # KubeRay setup and usage guide
├── config/                      # Configuration files
│   ├── prometheus.yml           # Prometheus scrape configuration
│   └── grafana/                 # Grafana provisioning
│       └── provisioning/
│           ├── dashboards/
│           │   ├── ray-dashboards.yml
│           │   └── json/        # Ray 2.54.0 Grafana dashboards (6 dashboards)
│           └── datasources/
│               └── prometheus.yml
├── helm/                        # Helm chart values
│   └── ray-cluster-values.yaml  # KubeRay cluster Helm values
├── scripts/                     # Shell scripts for cluster management
│   ├── common.sh                # Shared utilities and runtime detection
│   ├── kuberay-init.sh          # KubeRay cluster initialization
│   ├── kuberay-stop.sh          # KubeRay cluster shutdown
│   └── kuberay-status.sh        # KubeRay cluster status check
├── streamlit_app/               # Streamlit dashboard app
│   ├── app.py                   # Main Streamlit dashboard
│   └── jobs/                    # ML jobs for Ray execution
│       ├── mnist_training/      # MNIST example job
│       │   ├── train_mnist.py   # Training script
│       │   ├── runtime_env.yaml # Ray runtime environment (pip deps)
│       │   └── run.sh           # Job submission script
│       └── resnet_inference/    # ResNet example job
│           ├── inference.py     # Inference script
│           └── runtime_env.yaml # Ray runtime environment (pip deps)
├── hello_ray_job.py             # Simple Ray job example
├── ray_job_example.py           # Ray job submission example
├── docker-compose.yaml          # MinIO, Prometheus, Grafana orchestration
├── kind-config.yaml             # Kind cluster configuration
├── pyproject.toml               # Python project config (managed by uv)
├── uv.lock                      # Locked dependencies (committed to git)
├── Makefile                     # Simple commands for running services
├── .env.example                 # Environment variables template
├── LICENSE
└── README.md
```

---

## Components

### 1. **Ray 2.54.0 (via KubeRay)**
- **Purpose:** Distributed ML training, hyperparameter tuning, and job submission.
- **Deployment:** Kubernetes-based via Kind cluster and KubeRay operator.
- **Features:**
  - Production-like Kubernetes environment
  - Better resource management and scaling
  - Fault tolerance and high availability
  - REST API for job submission
  - Web dashboard for monitoring jobs and clusters
  - Runtime environment support for pip dependency management

### 2. **Streamlit**
- **Purpose:** Interactive dashboard for cluster status and S3 browsing.
- **Location:** [`streamlit_app/app.py`](streamlit_app/app.py)
- **Features:**
  - Check Ray cluster status and MinIO status
  - Browse and manage S3 buckets and files
  - Upload and download files from S3
  - Submit and monitor Ray jobs via UI
  - View system disk usage

### 3. **MinIO**
- **Purpose:** S3-compatible object storage system for local development.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml)
- **Default credentials:** minioadmin/minioadmin (configurable via `.env`)
- **Buckets:** app-bucket (public), ray-bucket
- **Ports:**
  - MinIO Server: 9000
  - Web Console: 9001

### 4. **Prometheus**
- **Purpose:** Metrics collection and time-series database for monitoring Ray cluster.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml), [`config/prometheus.yml`](config/prometheus.yml)
- **Features:**
  - Scrapes metrics from Ray on port 8080
  - Stores time-series data for historical analysis
  - Provides query interface for custom metrics
- **Port:** 9090

### 5. **Grafana**
- **Purpose:** Metrics visualization and dashboard for Ray cluster monitoring.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml)
- **Features:**
  - Pre-configured Prometheus datasource
  - 6 pre-built Ray 2.54.0 dashboards (Default, Data, Serve, Serve Deployment, Serve LLM, Train)
  - Customizable dashboards and alerts
- **Port:** 3000
- **Default credentials:** admin/admin

---

## Getting Started

> **Quick Start:** See [QUICKSTART.md](QUICKSTART.md) for a fast-track guide to get running in minutes!

### Prerequisites

- [Docker](https://www.docker.com/) or [Podman](https://podman.io/) (container runtime)
  - When using Podman, `podman-compose` is preferred and will be auto-installed via `uv tool install` if not present
- [Python](https://python.org/) 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

The following tools will be auto-installed via Homebrew if missing:
- [Kind](https://kind.sigs.k8s.io/) for local Kubernetes cluster
- [Helm](https://helm.sh/) for Kubernetes package management
- [kubectl](https://kubernetes.io/docs/tasks/tools/) for Kubernetes CLI

### Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd ml-homelab
   ```

2. **Configure environment variables (optional):**
   ```sh
   cp .env.example .env
   # Edit .env to customize credentials and ports
   # Note: .env is auto-created from .env.example on first `make start` if missing
   ```

3. **Install dependencies:**
   ```sh
   # uv creates the venv and installs everything automatically
   uv sync
   ```

4. **Start all services:**
   ```sh
   make start
   ```

   This will:
   - Verify `uv` and sync Python dependencies
   - Create a Kind Kubernetes cluster
   - Install KubeRay operator and Ray cluster
   - Start MinIO, Prometheus, and Grafana via Docker/Podman Compose
   - Set up port forwarding for Ray services
   - Start the Streamlit dashboard

5. **Check cluster status:**
   ```sh
   make status
   ```

6. **Access services:**
   - **Ray Dashboard:** http://localhost:8265/
   - **Streamlit Dashboard:** http://localhost:8501/
   - **MinIO Console:** http://localhost:9001/ (default: minioadmin/minioadmin)
   - **Prometheus:** http://localhost:9090/
   - **Grafana:** http://localhost:3000/ (default: admin/admin)

7. **Stop all services:**
   ```sh
   make stop
   ```

For detailed KubeRay setup instructions, see [docs/kuberay-setup.md](docs/kuberay-setup.md).

---

## Example Workflows

### Simple Ray Job

- Submit a job to the Ray cluster:
  ```sh
  make job SCRIPT=hello_ray_job.py
  ```

### Ray Job with Runtime Environment

- Jobs that need additional pip dependencies use `runtime_env.yaml` files:
  ```sh
  # ResNet inference (installs torch, torchvision, Pillow, numpy on the cluster)
  make job SCRIPT=streamlit_app/jobs/resnet_inference/inference.py \
       RUNTIME_ENV=streamlit_app/jobs/resnet_inference/runtime_env.yaml

  # MNIST training (installs torch, torchvision, filelock on the cluster)
  make job SCRIPT=streamlit_app/jobs/mnist_training/train_mnist.py \
       RUNTIME_ENV=streamlit_app/jobs/mnist_training/runtime_env.yaml
  ```

### Ray Job Submission (Programmatic)

- Submit a job and monitor its progress via Python API:
  ```sh
  uv run python ray_job_example.py
  ```

### Streamlit UI

- All jobs can also be submitted and monitored through the Streamlit dashboard:
  ```sh
  make run
  ```
  Use the Training and Inference tabs to submit jobs with automatic runtime environment handling.

### Architecture

```
+----------------+     +--------------+     +----------------+
|                |     |              |     |                |
|  Streamlit UI  +---->+  KubeRay     +---->+  ML Jobs      |
|  (Port 8501)   |     |  (Port 8265) |     |  (Training &  |
|                |     |              |     |   Inference)  |
+-------+--------+     +------+-------+     +-------+-------+
        |                     |                     |
        |                     | (metrics:8080)      |
        |                     v                     |
        |              +------+-------+             |
        |              |              |             |
        |              |  Prometheus  |             |
        |              |  (Port 9090) |             |
        |              +------+-------+             |
        |                     |                     |
        |                     v                     |
        |              +------+-------+             |
        |              |              |             |
        |              |   Grafana    |             |
        |              |  (Port 3000) |             |
        |              +--------------+             |
        |                                           |
        v                                           v
+-------+-------------------------------------------+-------+
|                                                           |
|                     MinIO Storage                         |
|                  (Ports 9000/9001)                        |
|                                                           |
+-----------------------------------------------------------+
```

- **Ray Client Server** (Port 10001): Allows job submissions via Ray client API
- **Ray Dashboard** (Port 8265): Provides monitoring and management UI for Ray jobs
- **Ray Metrics Export** (Port 8080): Exports Prometheus-compatible metrics
- **Prometheus** (Port 9090): Collects and stores time-series metrics from Ray
- **Grafana** (Port 3000): Visualizes Ray metrics with pre-built dashboards
- **MinIO Server** (Port 9000): S3-compatible API endpoint
- **MinIO Console** (Port 9001): Web UI for MinIO management

### Job Monitoring

You can monitor Ray jobs through:

1. **Streamlit Dashboard**: Shows real-time job status and logs
2. **Ray Dashboard**: http://localhost:8265/ - Detailed cluster and job metrics
3. **Grafana**: http://localhost:3000/ - Visual metrics dashboards for Ray cluster performance
4. **Prometheus**: http://localhost:9090/ - Query and explore raw metrics data
5. **Job Logs**: Available in the Streamlit UI under the Training/Inference tabs

### Metrics and Monitoring

The project includes a full metrics stack for monitoring Ray cluster performance:

#### Ray Metrics
Ray automatically exports metrics on port 8080, including:
- **System metrics**: CPU, memory, disk, and network usage per node
- **Application metrics**: Task execution times, actor lifecycle, object store usage
- **Job metrics**: Job submission, execution status, and resource utilization

#### Prometheus
Prometheus scrapes metrics from Ray every 15 seconds and stores them for historical analysis. Access the Prometheus UI at http://localhost:9090/ to:
- Run PromQL queries
- View metrics targets and their health
- Explore available metrics

#### Grafana
Grafana provides visual dashboards for Ray metrics at http://localhost:3000/ (admin/admin). Features include:
- Pre-configured Prometheus datasource
- 6 pre-built Ray 2.54.0 dashboards: Default, Data, Serve, Serve Deployment, Serve LLM, Train
- Customizable panels and alerts

### Streamlit Dashboard

- Use the dashboard to:
  - Check Ray and MinIO service status
  - View system disk usage
  - Browse and manage S3 buckets and files in MinIO
  - Upload and download files from S3 buckets
  - Submit and monitor Ray jobs for training and inference

---

## Testing

You can manually test your ML workflows through the Streamlit interface or by running the job scripts directly.

### Ray Job Testing

```sh
make job SCRIPT=hello_ray_job.py
```

### S3 Connection Testing

You can test S3 connectivity via the Streamlit interface or programmatically:

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

# List buckets
buckets = s3.list_buckets()
print([b["Name"] for b in buckets["Buckets"]])
```

---

## Customization

- **Add new ML experiments:**
  - Place scripts in [`streamlit_app/jobs/`](streamlit_app/jobs/)
  - Add a `runtime_env.yaml` listing pip dependencies needed on the Ray cluster
  - Follow the pattern in existing jobs (e.g., mnist_training, resnet_inference)
  - Update the Streamlit app to include new job types

- **Extend Streamlit UI:**
  - Edit [`streamlit_app/app.py`](streamlit_app/app.py)
  - Add new tabs, features, or integrations

- **Install extra Python packages:**
  ```sh
  uv add <package-name>
  ```

- **Configure Ray:**
  - Adjust parameters in the `.env` file
  - Modify Helm values in [`helm/ray-cluster-values.yaml`](helm/ray-cluster-values.yaml)

---

## Makefile

The Makefile provides convenient shortcuts:

```sh
make sync     # Sync Python dependencies with uv
make run      # Start the Streamlit app only
make start    # Start all services (KubeRay + Docker/Podman Compose)
make stop     # Stop all services
make status   # Check cluster status
make job SCRIPT=hello_ray_job.py                           # Submit a simple Ray job
make job SCRIPT=path/to/job.py RUNTIME_ENV=path/to/env.yaml  # Submit with pip deps
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [Ray](https://ray.io/) for distributed ML computation.
- [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) for Kubernetes-native Ray deployment.
- [Streamlit](https://streamlit.io/) for interactive dashboarding.
- [MinIO](https://min.io/) for S3-compatible object storage.
- [Prometheus](https://prometheus.io/) for metrics collection.
- [Grafana](https://grafana.com/) for metrics visualization.
- [uv](https://docs.astral.sh/uv/) for fast Python package management.
