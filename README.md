# ML Homelab

A local development environment for orchestrating, training, and visualizing machine learning workflows using Apache Airflow, Ray, and Streamlit. This project provides a reproducible setup for running ETL pipelines, distributed ML experiments, and interactive dashboards, with local emulation of AWS S3 via LocalStack.

---

## Project Structure

```
.
├── dags/                # Airflow DAGs (workflows)
│   └── exampledag.py    # Example astronaut ETL DAG
├── examples/
│   └── mnist_training/  # Example Ray Tune MNIST training job
│       ├── train_mnist.py
│       └── run.sh
├── streamlit_app/       # Streamlit dashboard app
│   ├── app.py
│   └── requirements.txt
├── tests/               # Pytest-based DAG and integration tests
│   └── dags/
├── .astro/              # Astro project config and DAG integrity tests
├── docker-compose.yaml  # Multi-service orchestration (Airflow, Ray, Streamlit, LocalStack)
├── Dockerfile           # Astro Runtime base image for Airflow
├── requirements.txt     # Python requirements for Airflow
├── packages.txt         # OS-level packages for Airflow image
├── airflow_settings.yaml# Local Airflow Connections/Variables/Pools
├── .env                 # Environment variables for Docker Compose
├── LICENSE
└── README.md
```

---

## Components

### 1. **Airflow**
- **Purpose:** Orchestrate ETL/data workflows.
- **Location:** [`dags/exampledag.py`](dags/exampledag.py)
- **Example DAG:** [`exampledag.py`](dags/exampledag.py) fetches and prints astronauts in space using the TaskFlow API and dynamic task mapping.

### 2. **Ray**
- **Purpose:** Distributed ML training and hyperparameter tuning.
- **Example:** [`examples/mnist_training/train_mnist.py`](streamlit_app/jobs/mnist_training/train_mnist.py) runs Ray Tune on MNIST with PyTorch.
- **How to run:** See [`examples/mnist_training/run.sh`](streamlit_app/jobs/mnist_training/run.sh).

### 3. **Streamlit**
- **Purpose:** Interactive dashboard for cluster status and S3 browsing.
- **Location:** [`streamlit_app/app.py`](streamlit_app/app.py)
- **Features:** 
  - Check Ray cluster status.
  - List S3 buckets via LocalStack.

### 4. **LocalStack**
- **Purpose:** Local AWS S3 emulation for development/testing.
- **Configured in:** [`docker-compose.yaml`](docker-compose.yaml)

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) (for Airflow development)

### Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd ml-homelab
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env` and adjust ports/services as needed.

3. **Start all services:**
   ```sh
   docker compose up
   ```
   This will launch Airflow, Ray (head/worker), Streamlit, and LocalStack.

4. **Access services:**
   - **Airflow UI:** http://localhost:8080/ (default, if using Astro CLI)
   - **Streamlit Dashboard:** http://localhost:8501/
   - **Ray Dashboard:** http://localhost:8265/
   - **LocalStack S3:** http://localhost:4566/

---

## Example Workflows

### Airflow DAG

- See [`dags/exampledag.py`](dags/exampledag.py) for a sample ETL pipeline using the Open Notify API.

### Ray MNIST Training

- Run distributed MNIST training with Ray Tune:
  ```sh
  cd jobs/mnist_training
  ./run.sh
  ```

### Streamlit Dashboard

- Use the dashboard to check Ray cluster status and list S3 buckets (via LocalStack).

---

## Testing

- **DAG Integrity:** Custom and Astro-provided tests in [`tests/dags/`](tests/dags/) and [`.astro/test_dag_integrity_default.py`](.astro/test_dag_integrity_default.py).
- **Run tests:**
  ```sh
  pytest tests/
  ```

---

## Customization

- **Add new Airflow DAGs:** Place Python files in [`dags/`](dags/).
- **Add new ML experiments:** Place scripts in [`examples/`](streamlit_app/jobs/).
- **Extend Streamlit UI:** Edit [`streamlit_app/app.py`](streamlit_app/app.py).
- **Install extra Python packages:** Add to [`requirements.txt`](requirements.txt) or [`streamlit_app/requirements.txt`](streamlit_app/requirements.txt).
- **Install OS packages:** Add to [`packages.txt`](packages.txt).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [Astronomer](https://www.astronomer.io/) for Airflow runtime and project scaffolding.
- [Ray](https://ray.io/) for distributed ML.
- [Streamlit](https://streamlit.io/) for dashboarding.
- [LocalStack](https://localstack.cloud/) for AWS emulation.