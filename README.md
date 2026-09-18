# ML Homelab

A local development environment for orchestrating, training, and visualizing machine learning workflows using KubeRay and Streamlit. This project provides a reproducible setup for running distributed ML experiments on Kubernetes, with S3-compatible storage via Floci and metrics monitoring via Prometheus and Grafana.

---

## Project Structure

```
.
├── data/                        # Data storage for services (auto-created)
│   ├── floci/                   # Floci S3-compatible storage
│   ├── prometheus/              # Prometheus time-series data
│   └── grafana/                 # Grafana configuration data
├── docs/                        # Documentation
│   ├── kuberay-setup.md         # KubeRay setup and usage guide
│   └── multi-machine.md         # Multi-machine Ray cluster guide
├── config/                      # Configuration files
│   ├── prometheus.yml           # Prometheus scrape configuration
│   ├── floci/                   # Floci provisioning
│   │   └── init/ready.d/        # Startup hooks (creates the S3 buckets)
│   └── grafana/                 # Grafana provisioning
│       └── provisioning/
│           ├── dashboards/
│           │   ├── ray-dashboards.yml
│           │   └── json/        # Ray 2.58.0 Grafana dashboards (8, generated)
│           └── datasources/
│               └── prometheus.yml
├── helm/                        # Helm chart values
│   └── ray-cluster-values.yaml  # KubeRay cluster Helm values
├── scripts/                     # Shell scripts for cluster management
│   ├── common.sh                # Shared utilities and runtime detection
│   ├── services.sh              # Compose services (Floci, Prometheus, Grafana)
│   ├── streamlit.sh             # Streamlit dashboard start/stop
│   ├── kuberay-init.sh          # KubeRay cluster initialization
│   ├── kuberay-stop.sh          # KubeRay cluster shutdown
│   ├── kuberay-status.sh        # KubeRay cluster status check
│   ├── ray-node-setup.sh        # Prepare a machine to run a Ray node
│   ├── ray-head-start.sh        # Native Ray head (multi-machine topology)
│   ├── ray-worker-start.sh      # Join a machine to a Ray head
│   └── ray-node-stop.sh         # Stop this machine's Ray node
├── streamlit_app/               # Streamlit dashboard app
│   ├── app.py                   # Streamlit UI (rendering only)
│   ├── health.py                # Service health and disk checks
│   ├── storage.py               # Floci/S3 helpers
│   ├── job_runner.py            # Ray Jobs API submission and polling
│   └── jobs/                    # ML jobs for Ray execution
│       ├── mnist_training/      # MNIST example job
│       │   ├── train_mnist.py   # Training script
│       │   ├── runtime_env.yaml # Ray runtime environment (pip deps)
│       │   └── run.sh           # Job submission script
│       ├── resnet_inference/    # ResNet example job
│       │   ├── inference.py     # Inference script
│       │   └── runtime_env.yaml # Ray runtime environment (pip deps)
│       └── memray_profiling/    # Memray worker-profiling job
│           ├── workload.py      # Synthetic pipeline both examples profile
│           ├── profile_job.py   # Profiles Ray workers with memray
│           ├── runtime_env.yaml # Ray runtime environment (pip deps)
│           └── run.sh           # Job submission script
├── tests/                       # Pytest suite (see Testing below)
├── .github/workflows/ci.yml     # Lint, type check, and test on every push
├── examples/                    # Standalone Ray examples
│   ├── hello_ray_job.py         # Simple Ray job
│   ├── memray_example.py        # Local memray profile + flamegraph views
│   └── ray_job_example.py       # Job submission via the Ray Jobs API
├── docker-compose.yaml          # Floci, Prometheus, Grafana orchestration
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

### 1. **Ray 2.58.0 (via KubeRay)**
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
  - Check Ray cluster status and Floci status
  - Browse and manage S3 buckets and files
  - Upload and download files from S3
  - Submit and monitor Ray jobs via UI
  - View system disk usage

### 3. **Floci**
- **Purpose:** S3-compatible object storage for local development. Floci is a
  local AWS emulator; only its S3 service is used here.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml)
- **Credentials:** any non-empty pair works (`.env` ships `test`/`test`)
- **Buckets:** app-bucket (public), ray-bucket — created on boot by
  [`config/floci/init/ready.d/01-create-buckets.sh`](config/floci/init/ready.d/01-create-buckets.sh)
- **Storage:** `FLOCI_STORAGE_MODE=persistent`, backed by `data/floci/`
- **Port:** 4566 (S3 API), published on `127.0.0.1` only. Floci accepts any
  non-empty credential and does not verify signatures, so binding it to all
  interfaces would expose every bucket to the network; override with
  `FLOCI_BIND_HOST` only on a network you trust.
- Floci is API-only — there is no web console; browse buckets from the
  Streamlit dashboard's S3 tab.

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
  - 8 pre-built Ray 2.58.0 dashboards (Default, Data, Data LLM, Serve,
    Serve Deployment, Serve LLM, Serve LLM SGLang, Train), exported with
    `make dashboards`
  - Customizable dashboards and alerts
- **Port:** 3000
- **Default credentials:** admin/admin

---

## Getting Started

> **Quick Start:** See [QUICKSTART.md](QUICKSTART.md) for a fast-track guide to get running in minutes!

### Prerequisites

- [Docker](https://www.docker.com/) or [Podman](https://podman.io/) (container runtime)
  - When using Podman, `podman-compose` is preferred and will be auto-installed via `uv tool install` if not present
- [Python](https://python.org/) 3.12 (3.13 is not yet supported)
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
   - Start Floci, Prometheus, and Grafana via Docker/Podman Compose
   - Set up port forwarding for Ray services
   - Start the Streamlit dashboard

5. **Check cluster status:**
   ```sh
   make status
   ```

6. **Access services:**
   - **Ray Dashboard:** http://localhost:8265/
   - **Streamlit Dashboard:** http://localhost:8501/
   - **Floci S3 API:** http://localhost:4566/ (no console — use the Streamlit S3 tab)
   - **Prometheus:** http://localhost:9090/
   - **Grafana:** http://localhost:3000/ (default: admin/admin)

7. **Stop all services:**
   ```sh
   make stop
   ```

### Make targets

The targets are grouped so each piece can be driven on its own.

| Group | Targets |
|---|---|
| Services (Floci, Prometheus, Grafana) | `services-up`, `services-down`, `services-status`, `services-logs` |
| Floci alone | `floci-up`, `floci-down`, `floci-logs` |
| Streamlit | `app` (background), `app-stop`, `run` (foreground) |
| KubeRay topology | `kuberay-start`, `kuberay-stop`, `kuberay-status` |
| Standalone Ray topology | `ray-setup`, `ray-head`, `ray-worker HEAD=<ip>`, `ray-stop`, `ray-status` |
| Development | `sync`, `test`, `cov`, `lint`, `format`, `typecheck`, `check`, `hooks`, `dashboards` |

`start`, `stop`, and `status` remain aliases for the KubeRay targets.

### Two topologies

- **KubeRay** (`make start`) — one machine, Ray on Kubernetes via Kind, with
  Prometheus and Grafana wired into the Ray dashboard. This is the default.
- **Standalone Ray** (`make ray-head` / `make ray-worker`) — several machines
  on a LAN running Ray natively, no Kubernetes. Every node syncs the same
  pinned environment from this repo so the Ray and Python versions match
  exactly. See [docs/multi-machine.md](docs/multi-machine.md).

Run one or the other; they contend for ports 6379 and 8265.

For detailed KubeRay setup instructions, see [docs/kuberay-setup.md](docs/kuberay-setup.md).

---

## Example Workflows

### Simple Ray Job

- Submit a job to the Ray cluster:
  ```sh
  make job SCRIPT=examples/hello_ray_job.py
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
  uv run python -m examples.ray_job_example
  ```

### Memory Profiling with memray

Ray's metrics stack shows *that* a worker's memory grew;
[memray](https://github.com/bloomberg/memray) shows *which Python code
allocated it*. Both examples profile the same pipeline, in
[`workload.py`](streamlit_app/jobs/memray_profiling/workload.py), whose call
tree is shaped to make a flamegraph worth reading.

Profile locally and render all three views:

```sh
make memray
```

This writes `profile.bin` plus three HTML reports to `.memray/`. Open
`memray-flamegraph-profile.html` first.

#### Reading the flamegraph

A memray flamegraph is **not** a timeline. Nothing about the x-axis means
"later" — bars are sorted alphabetically, not chronologically. What the axes
actually mean:

- **Width** is bytes allocated. Wider means more memory, nothing else.
- **Height** is stack depth. A bar sits directly on top of whatever called it.
- **A bar's width is the sum of its children**, so a wide bar with one wide
  child is just a pass-through; the allocation happened further up.

Frames are merged *by stack*, not by function. The same function reached from
two different callers appears as two separate bars rather than being pooled
into one. `_decode` is the example here — it is called from both
`RecordBuffer.append` and `transform_rows`, and shows up under each of them
separately (288 KB and 4.5 KB respectively). You will only see it once you turn
on Python-allocator tracing; see "Why some functions are missing" below.

The pipeline is built from three tiers so each view shows something different:

```
run_workload
├── ingest_batches        256.0 MB   retained — never freed
│   ├── RecordBuffer.append → _decode
│   └── RecordBuffer.compact
├── build_index            22.8 MB   retained — a 7-frame recursion tower
│   └── _index_node → _index_node → …
├── transform_rows         64.0 MB   live at peak, freed on return
│   ├── _decode            ← same leaf, second parent
│   └── _widen
└── summarize_rows                   churn only
    └── _fold
```

`build_index` is the one to look at first: because `_index_node` calls itself,
it draws a narrow tower seven frames tall. Recursion is unmistakable once you
know that shape.

#### The three views

Each renders the same capture, selecting different allocations. Measured at the
defaults:

| frame | peak (default) | `--leaks` | `--temporary-allocations` |
| --- | ---: | ---: | ---: |
| `ingest_batches` → `compact` | 256.0 MB | 256.0 MB | 2.5 MB |
| `build_index` → `_index_node` | 22.8 MB | 22.8 MB | — |
| `transform_rows` → `_widen` | 64.0 MB | — | 2.0 MB |

- **peak** (`memray-flamegraph-profile.html`) — what was live at the high
  watermark. The three stages sum to the 343 MB peak that `make memray` prints.
- **leaks** (`memray-leaks-profile.html`) — what was never freed. The 64 MB
  `transform_rows` block **disappears**: it was live at peak but released on
  return. Only the two retaining stages survive. This is the view for hunting a
  real leak, and here it points straight at the module-level `CACHE`.
- **temporary** (`memray-temporary-profile.html`) — allocations freed almost
  immediately. The big retained blocks shrink to the churn that produced them.

Render just one with `--view`:

```sh
uv run python -m examples.memray_example --view leaks
```

#### Why some functions are missing

`_decode`, `RecordBuffer.append`, and `_fold` allocate on every call, yet none
of them appear in any view above. They are not being hidden — memray traces
calls into the system allocator, and CPython serves small objects from pymalloc
pools it has already claimed. No `malloc`, no record.

Trace pymalloc itself to get them back:

```sh
uv run python -m examples.memray_example --trace-python-allocators
```

That takes the same run from 68,066 recorded allocations to 423,941 — a 6x
jump, all of it small objects — and `_decode`, `append`, and `_fold` appear in
the temporary view. The cost is a much larger capture and a slower run, which
is why it is off by default. Reach for it when a function you *know* allocates
is missing from the graph.

### Profiling Ray Workers

`memray.Tracker` profiles the process it runs in, so wrapping the driver would
only measure the process handing out work. The
[profiling job](streamlit_app/jobs/memray_profiling/profile_job.py) opens the
tracker **inside the Ray task** instead, renders each capture to a
self-contained HTML report in the worker, and returns the bytes over the object
store for the driver to write out:

```sh
make job SCRIPT=streamlit_app/jobs/memray_profiling/profile_job.py \
     RUNTIME_ENV=streamlit_app/jobs/memray_profiling/runtime_env.yaml
```

The job log gets one row per worker, plus a flamegraph per task:

```
     pid          peak    allocations      retained
---------------------------------------------------
   11882      200.1 MB         34,042      135.6 MB
   11881      200.1 MB         34,042      135.6 MB

Profiled 2 task(s) across 2 worker process(es).

  flamegraph: .memray-ray/memray-worker-11882-task0.html
  flamegraph: .memray-ray/memray-worker-11881-task1.html
```

Distinct pids are the evidence the tracker ran in the workers rather than the
driver. Where those HTML files land depends on where the driver runs:

- **Local Ray** (`uv run python streamlit_app/jobs/memray_profiling/profile_job.py`)
  — the driver is on your machine, so `--output-dir` is a local directory and
  you can open the reports directly.
- **Submitted to KubeRay** — the driver runs in a pod, so copy them out:
  ```sh
  kubectl cp <ray-worker-pod>:/home/ray/.memray-ray ./.memray-ray
  ```

#### Viewing profiles from the Ray dashboard

Ray has its own memray integration, which is usually the faster way to look at
a live worker. It is enabled here via `RAY_DASHBOARD_ENABLE_PROFILING=1` in
[`helm/ray-cluster-values.yaml`](helm/ray-cluster-values.yaml) (and in
[`scripts/kuberay-init.sh`](scripts/kuberay-init.sh), which rewrites the head's
env at deploy time).

Open the Ray dashboard at http://localhost:8265/, find a worker, actor, task,
or job driver, and use its **Memory profiling** action. You can pick the
format (flamegraph or table), a duration, and the same `--leaks` / native /
Python-allocator options described above; the report renders in the browser.

> The dashboard runs `memray` from the *node's* environment, not the job's
> runtime env, so the Ray image needs it too. The image does not ship memray —
> install it into a running pod with
> `kubectl exec <pod> -- pip install memray`, or bake it into a custom image.

> Profiling exposes side-effecting endpoints. It is enabled here only because
> the dashboard is reached over a local port-forward; do not enable it on a
> dashboard published to the network without token authentication.

> memray supports Linux and macOS 11+ only; it has no Windows build.

### Streamlit UI

- All jobs can also be submitted and monitored through the Streamlit dashboard:
  ```sh
  make run
  ```
  Use the Training, Inference, and Profiling tabs to submit jobs with automatic runtime environment handling.

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
|                     Floci Storage                         |
|                      (Port 4566)                          |
|                                                           |
+-----------------------------------------------------------+
```

- **Ray Client Server** (Port 10001): Allows job submissions via Ray client API
- **Ray Dashboard** (Port 8265): Provides monitoring and management UI for Ray jobs
- **Ray Metrics Export** (Port 8080): Exports Prometheus-compatible metrics
- **Prometheus** (Port 9090): Collects and stores time-series metrics from Ray
- **Grafana** (Port 3000): Visualizes Ray metrics with pre-built dashboards
- **Floci** (Port 4566): S3-compatible API endpoint

### Job Monitoring

You can monitor Ray jobs through:

1. **Streamlit Dashboard**: Shows real-time job status and logs
2. **Ray Dashboard**: http://localhost:8265/ - Detailed cluster and job metrics
3. **Grafana**: http://localhost:3000/ - Visual metrics dashboards for Ray cluster performance
4. **Prometheus**: http://localhost:9090/ - Query and explore raw metrics data
5. **Job Logs**: Available in the Streamlit UI under the Training/Inference/Profiling tabs

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
- 8 pre-built Ray 2.58.0 dashboards: Default, Data, Data LLM, Serve, Serve Deployment, Serve LLM, Serve LLM SGLang, Train
- Customizable panels and alerts

### Streamlit Dashboard

- Use the dashboard to:
  - Check Ray and Floci service status
  - View system disk usage
  - Browse and manage S3 buckets and files in Floci
  - Upload and download files from S3 buckets
  - Submit and monitor Ray jobs for training, inference, and memory profiling

---

## Testing

The project uses **pytest** with **coverage**, **ruff** for linting and formatting,
and **ty** for type checking — all pinned in `uv.lock` and run through `uv run`.

```sh
make test        # run the test suite
make cov         # run with coverage (term + HTML report in htmlcov/)
make lint        # ruff check
make format      # ruff format + autofix
make typecheck   # ty check
make check       # everything CI runs
make hooks       # install the pre-commit and pre-push git hooks
```

### PyTorch builds (CPU vs GPU)

`torch` is not in `dependencies`. It lives in two mutually exclusive extras, so
one `uv.lock` carries both builds:

| extra | on Linux | on macOS |
| --- | --- | --- |
| `cpu` | `torch 2.13.0+cpu` from `download.pytorch.org/whl/cpu` | PyPI wheel (CPU/MPS) |
| `gpu` | `torch 2.13.0+cu129` from `download.pytorch.org/whl/cu129` | PyPI wheel (CPU/MPS) |

Hardware cannot be expressed as a dependency marker, so the choice happens at
sync time rather than in the lockfile. [`scripts/torch-extra.sh`](scripts/torch-extra.sh)
probes `nvidia-smi` and prints `gpu` or `cpu`, and the Makefile passes that to
every uv command:

```sh
make sync                    # picks the build matching this machine
make sync TORCH_EXTRA=gpu    # force the CUDA build
make test TORCH_EXTRA=cpu    # force the CPU build
```

CI pins `--extra cpu` explicitly, since GitHub runners have no GPU and the CUDA
wheels add roughly 3 GB.

> **One gotcha:** because torch is an extra, a bare `uv run pytest` syncs an
> environment *without* it and the tests fail to import. Use `make test`, or
> [`scripts/uv-run.sh`](scripts/uv-run.sh), or pass `--extra` yourself.

Ray generates its Grafana dashboards from code, so the committed JSON goes
stale whenever Ray is upgraded. After a version bump, run:

```sh
make dashboards
```

A test asserts the committed dashboards match the installed Ray version, so
CI fails rather than letting them drift.

Coverage is configured in `pyproject.toml` and fails below **95%**. The suite
covers the health checks, the S3 helpers, runtime-environment assembly and job
polling, all three Ray job scripts, and the Streamlit UI itself via
`streamlit.testing.v1.AppTest` — no running cluster, Floci, or network is
required.

### Ray Job Testing

Against a real cluster:

```sh
make job SCRIPT=examples/hello_ray_job.py
```

### S3 Connection Testing

You can test S3 connectivity via the Streamlit interface or programmatically:

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
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
  - Put rendering in [`streamlit_app/app.py`](streamlit_app/app.py) and logic in
    `health.py`, `storage.py`, or `job_runner.py` so it stays unit-testable
  - Add a matching test under [`tests/`](tests/)

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
make job SCRIPT=examples/hello_ray_job.py                           # Submit a simple Ray job
make job SCRIPT=path/to/job.py RUNTIME_ENV=path/to/env.yaml  # Submit with pip deps
make memray   # Profile a synthetic pipeline and render all three flamegraph views
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [Ray](https://ray.io/) for distributed ML computation.
- [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) for Kubernetes-native Ray deployment.
- [Streamlit](https://streamlit.io/) for interactive dashboarding.
- [Floci](https://floci.io/) for S3-compatible object storage.
- [Prometheus](https://prometheus.io/) for metrics collection.
- [Grafana](https://grafana.com/) for metrics visualization.
- [memray](https://github.com/bloomberg/memray) for Python memory profiling.
- [uv](https://docs.astral.sh/uv/) for fast Python package management.
