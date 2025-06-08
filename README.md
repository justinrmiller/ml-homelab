# ML Homelab

A local development environment for orchestrating, training, and visualizing machine learning workflows using Apache Airflow, Ray, and Streamlit. This project provides a reproducible setup for running ETL pipelines, distributed ML experiments, and interactive dashboards, with local emulation of AWS S3 via Minio.

---

## Project Structure

```
.
├── data/                        # Data storage for services
│   ├── minio/                   # MinIO S3-compatible storage
│   │   ├── app-bucket/          # General application bucket
│   │   └── ray-bucket/          # Ray-specific bucket
├── streamlit_app/               # Streamlit dashboard app
│   ├── app.py                   # Main Streamlit dashboard
│   ├── requirements.txt         # Python dependencies for Streamlit
│   └── jobs/                    # ML jobs for Ray execution
│       ├── mnist_training/      # MNIST example job
│       │   ├── train_mnist.py   # Training script
│       │   └── run.sh           # Job submission script
│       └── resnet_inference/    # ResNet example job
│           └── inference.py     # Inference script
├── docker-compose.yaml          # MinIO orchestration
├── Makefile                     # Simple commands for running services
├── .env                         # Environment variables for services
├── LICENSE
└── README.md
```

---

## Components

### 1. **Ray**
- **Purpose:** Distributed ML training and hyperparameter tuning.

### 2. **Streamlit**
- **Purpose:** Interactive dashboard for cluster status and S3 browsing.
- **Location:** [`streamlit_app/app.py`](streamlit_app/app.py)
- **Features:**
  - Check Ray cluster status.
  - List S3 buckets and manage files.
  - Submit Ray jobs via UI.

### 3. **MinIO**
- **Purpose:** S3-compatible object storage system for local development.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml)
- **Buckets:** app-bucket (public), ray-bucket

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html) for distributed ML
- [Streamlit](https://streamlit.io/) for the dashboard

### Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd ml-homelab
   ```

2. **Configure environment variables:**
   - Check and adjust `.env` file for your environment.

3. **Install dependencies:**
   ```sh
   pip install -r streamlit_app/requirements.txt
   ```

4. **Start all services at once:**

   Using the convenience commands in the Makefile:
   ```sh
   make start    # Start all services (MinIO, Ray, Streamlit)
   make stop     # Stop all services
   ```

   This will:
   - Start MinIO using Docker Compose
   - Launch Ray as a head node
   - Start the Streamlit dashboard

   Alternatively, you can directly use the scripts:
   ```sh
   ./init.sh     # Start all services
   ./stop.sh     # Stop all services
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
     --dashboard-port=8265
   ```

   Start Streamlit dashboard (in a separate terminal):
   ```sh
   make run
   ```

5. **Access services:**
   - **MinIO Console:** http://localhost:9001/ (credentials from .env)
   - **Streamlit Dashboard:** http://localhost:8501/
   - **Ray Dashboard:** http://localhost:8265/

---

## Example Workflows

### Ray MNIST Training

- Run distributed MNIST training with Ray Tune:
  ```sh
  cd streamlit_app/jobs/mnist_training
  ./run.sh
  ```

- Or submit the job through the Streamlit UI using the Training tab.

### Streamlit Dashboard

- Use the dashboard to:
  - Check Ray cluster status
  - Browse and manage S3 buckets and files in MinIO
  - Submit Ray jobs for training and inference

---

## Testing

This project doesn't currently include automated tests. You can manually test your ML workflows through the Streamlit interface or by running the job scripts directly.

---

## Customization

- **Add new ML experiments:** Place scripts in [`streamlit_app/jobs/`](streamlit_app/jobs/).
- **Extend Streamlit UI:** Edit [`streamlit_app/app.py`](streamlit_app/app.py).
- **Install extra Python packages:** Add to [`streamlit_app/requirements.txt`](streamlit_app/requirements.txt).
- **Configure Ray:** Adjust parameters in the Ray start command or in the run scripts.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [Ray](https://ray.io/) for distributed ML computation.
- [Streamlit](https://streamlit.io/) for interactive dashboarding.
- [MinIO](https://min.io/) for S3-compatible object storage.
