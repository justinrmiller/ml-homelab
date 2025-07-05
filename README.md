# ML Homelab

A local development environment for orchestrating, training, and visualizing machine learning workflows using Ray and Streamlit. This project provides a reproducible setup for running distributed ML experiments and interactive dashboards, with local emulation of AWS S3 via MinIO.

---

## Project Structure

```
.
├── data/                        # Data storage for services
│   ├── minio/                   # MinIO S3-compatible storage
│   │   ├── app-bucket/          # General application bucket
│   │   └── ray-bucket/          # Ray-specific bucket
├── docs/                        # Documentation
│   ├── docker-setup.md          # Docker and Docker Compose guide
│   └── kuberay-setup.md         # KubeRay setup and usage guide
├── scripts/                     # Shell scripts for cluster management
│   ├── init.sh                  # Script to start all services
│   ├── stop.sh                  # Script to stop all services
│   ├── kuberay-init.sh          # KubeRay cluster initialization
│   ├── kuberay-stop.sh          # KubeRay cluster shutdown
│   └── kuberay-status.sh        # KubeRay cluster status check
├── streamlit_app/               # Streamlit dashboard app
│   ├── app.py                   # Main Streamlit dashboard
│   ├── requirements.txt         # Python dependencies for Streamlit
│   └── jobs/                    # ML jobs for Ray execution
│       ├── mnist_training/      # MNIST example job
│       │   ├── train_mnist.py   # Training script
│       │   └── run.sh           # Job submission script
│       └── resnet_inference/    # ResNet example job
│           └── inference.py     # Inference script
├── hello_ray_job.py             # Simple Ray job example
├── ray_job_example.py           # Ray job submission example
├── docker-compose.yaml          # MinIO orchestration
├── Makefile                     # Simple commands for running services
├── .env.example                 # Environment variables template
├── LICENSE
└── README.md
```

---

## Components

### 1. **Ray**
- **Purpose:** Distributed ML training, hyperparameter tuning, and job submission.
- **Version:** 2.46.0
- **Features:**
  - Distributed computing framework for machine learning workloads
  - REST API for job submission
  - Web dashboard for monitoring jobs and clusters

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
- **Buckets:** app-bucket (public), ray-bucket
- **Ports:**
  - MinIO Server: 9000
  - Web Console: 9001

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html) for distributed ML
- [Streamlit](https://streamlit.io/) for the dashboard
- [Python](https://python.org/) - version 3.10/3.11

### KubeRay Prerequisites (Optional)

For KubeRay setup, you'll also need:
- [Kind](https://kind.sigs.k8s.io/) for local Kubernetes cluster
- [Helm](https://helm.sh/) for Kubernetes package management
- [kubectl](https://kubernetes.io/docs/tasks/tools/) for Kubernetes CLI

### Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd ml-homelab
   ```

2. **Configure environment variables:**
   - Check and adjust `.env` file for your environment:
     - MinIO credentials (default: test-key/test-secret)
     - Ray configuration settings
     - Architecture-specific Ray image selection
   ```
    # Key settings in .env:
    AWS_ACCESS_KEY_ID=test-key
    AWS_SECRET_ACCESS_KEY=test-secret
    MINIO_ROOT_USER=test-key
    MINIO_ROOT_PASSWORD=test-secret

    # Ray settings
    DASHBOARD_PORT=8265
    HEAD_NODE_PORT=10001
    PORT=6379
    RAY_ADDRESS=ray://127.0.0.1:10001
   ```

3. **Install dependencies:**
   ```sh
   pip install -r streamlit_app/requirements.txt
   pip install ray[default]==2.47.0
   ```

4. **Start all services at once:**

   Using the convenience commands in the Makefile:
   ```sh
   make start    # Start all services (MinIO, Ray, Streamlit)
   make stop     # Stop all services
   ```

   This will:
   - Start MinIO using Docker Compose
   - Launch Ray as a head node with client server port 10001
   - Start the Streamlit dashboard

   Alternatively, you can directly use the scripts:
   ```sh
   scripts/init.sh     # Start all services
   scripts/stop.sh     # Stop all services
   ```

   **Alternatively, start services separately:**

   Start MinIO (S3 storage):
   ```sh
   docker compose up -d
   ```

   Start Ray (in a separate terminal):
   ```sh
   ray start --head \
     --port=6379 \
     --dashboard-port=8265 \
     --ray-client-server-port=10001 \
     --node-ip-address=127.0.0.1 \
     --dashboard-host=127.0.0.1
   ```

   Start Streamlit dashboard (in a separate terminal):
   ```sh
   make run
   ```

5. **Access services:**
   - **MinIO Console:** http://localhost:9001/ (credentials from .env)
   - **Streamlit Dashboard:** http://localhost:8501/
   - **Ray Dashboard:** http://localhost:8265/

### KubeRay Setup (Alternative)

For a more production-like setup using Kubernetes and KubeRay:

1. **Start KubeRay cluster:**
   ```sh
   # Using Makefile
   make kuberay-start
   
   # Or using scripts directly
   scripts/kuberay-init.sh
   
   # Or using unified interface
   scripts/init.sh --kuberay
   ```

2. **Check cluster status:**
   ```sh
   # Using Makefile
   make kuberay-status
   
   # Or using script directly
   scripts/kuberay-status.sh
   ```

3. **Stop KubeRay cluster:**
   ```sh
   # Using Makefile
   make kuberay-stop
   
   # Or using scripts directly
   scripts/kuberay-stop.sh
   
   # Or using unified interface
   scripts/stop.sh --kuberay
   ```

4. **Access services** (same URLs as local setup):
   - **MinIO Console:** http://localhost:9001/
   - **Streamlit Dashboard:** http://localhost:8501/
   - **Ray Dashboard:** http://localhost:8265/

**KubeRay Benefits:**
- Production-like Kubernetes environment
- Better resource management and scaling
- Fault tolerance and high availability
- Supports multi-tenancy

For detailed KubeRay setup instructions, see [docs/kuberay-setup.md](docs/kuberay-setup.md).

---

## Example Workflows

### Simple Ray Job

- Run a simple Ray job directly:
  ```sh
  python hello_ray_job.py
  ```

### Ray Job Submission

- Submit a job to the Ray cluster and monitor its progress:
  ```sh
  python ray_job_example.py
  ```
  This will submit and monitor the execution of `hello_ray_job.py`.

### Ray MNIST Training

- Run distributed MNIST training with Ray Tune:
  ```sh
  cd streamlit_app/jobs/mnist_training
  ./run.sh
  ```

- Or submit the job through the Streamlit UI using the Training tab.

### ResNet Inference

- Run inference using a pre-trained ResNet model:
  ```sh
  cd streamlit_app/jobs/resnet_inference
  python inference.py
  ```

- Or submit the job through the Streamlit UI using the Inference tab.

### Architecture

The ML Homelab uses the following architecture:

```
+----------------+     +--------------+     +----------------+
|                |     |              |     |                |
|  Streamlit UI  +---->+  Ray Cluster +---->+  ML Jobs      |
|  (Port 8501)   |     |  (Port 8265) |     |  (Training &  |
|                |     |              |     |   Inference)  |
+-------+--------+     +------+-------+     +-------+-------+
        |                     |                     |
        |                     |                     |
        v                     v                     v
+-------+---------------------+---------------------+-------+
|                                                           |
|                     MinIO Storage                         |
|                  (Ports 9000/9001)                        |
|                                                           |
+-----------------------------------------------------------+
```

- **Ray Client Server** (Port 10001): Allows job submissions via Ray client API
- **Ray Dashboard** (Port 8265): Provides monitoring and management UI for Ray jobs
- **MinIO Server** (Port 9000): S3-compatible API endpoint
- **MinIO Console** (Port 9001): Web UI for MinIO management

### Job Monitoring

You can monitor Ray jobs through:

1. **Streamlit Dashboard**: Shows real-time job status and logs
2. **Ray Dashboard**: http://localhost:8265/ - Detailed cluster and job metrics
3. **Job Logs**: Available in the Streamlit UI under the Training/Inference tabs

### Streamlit Dashboard

- Use the dashboard to:
  - Check Ray and MinIO service status
  - View system disk usage
  - Browse and manage S3 buckets and files in MinIO
  - Upload and download files from S3 buckets
  - Submit and monitor Ray jobs for training and inference

---

## Testing

This project doesn't currently include automated tests. You can manually test your ML workflows through the Streamlit interface or by running the job scripts directly.

### Ray Job Testing

You can test the Ray setup with the simple hello_ray_job.py script:

```sh
python hello_ray_job.py
```

Expected output: "hello world"

### S3 Connection Testing

You can test S3 connectivity via the Streamlit interface or programmatically:

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="test-key",
    aws_secret_access_key="test-secret",
)

# List buckets
buckets = s3.list_buckets()
print([b["Name"] for b in buckets["Buckets"]])
```

---

## Customization

- **Add new ML experiments:**
  - Place scripts in [`streamlit_app/jobs/`](streamlit_app/jobs/)
  - Follow the pattern in existing jobs (e.g., mnist_training, resnet_inference)
  - Update the Streamlit app to include new job types

- **Extend Streamlit UI:**
  - Edit [`streamlit_app/app.py`](streamlit_app/app.py)
  - Add new tabs, features, or integrations
  - Current structure uses tabs for S3 browsing, Training jobs, and Inference jobs

- **Install extra Python packages:**
  - Add to [`streamlit_app/requirements.txt`](streamlit_app/requirements.txt)
  - Install using `pip install -r streamlit_app/requirements.txt`

- **Configure Ray:**
  - Adjust parameters in the `.env` file
  - Modify the Ray start command in `init.sh`
  - Add custom Ray configuration in job scripts

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Helper Scripts

### scripts/init.sh

The `scripts/init.sh` script automates the startup of all services:
- Checks for prerequisites (Docker, Ray, Streamlit)
- Starts MinIO using Docker Compose
- Starts Ray as a head node with the client server
- Starts the Streamlit application
- Provides status checks and URLs for each service
- Supports `--kuberay` flag for KubeRay mode

Usage:
```sh
scripts/init.sh           # Local Ray setup
scripts/init.sh --kuberay # KubeRay setup
```

### scripts/stop.sh

The `scripts/stop.sh` script gracefully shuts down all services:
- Stops the Streamlit application
- Stops the Ray cluster
- Stops MinIO Docker containers
- Supports `--kuberay` flag for KubeRay cleanup

Usage:
```sh
scripts/stop.sh           # Stop local Ray setup
scripts/stop.sh --kuberay # Stop KubeRay setup
```

### Makefile

The Makefile provides convenient shortcuts:

**Local Ray Setup:**
```sh
make run    # Start the Streamlit app only
make start  # Start all services (calls init.sh)
make stop   # Stop all services (calls stop.sh)
```

**KubeRay Setup:**
```sh
make kuberay-start   # Start KubeRay cluster
make kuberay-status  # Check KubeRay cluster status
make kuberay-stop    # Stop KubeRay cluster
```

---

## Credits

- [Ray](https://ray.io/) for distributed ML computation.
- [Streamlit](https://streamlit.io/) for interactive dashboarding.
- [MinIO](https://min.io/) for S3-compatible object storage.
