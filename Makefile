.PHONY: run start stop setup-metrics kuberay-start kuberay-stop kuberay-status

# Start Streamlit app only
run:
	uv run streamlit run streamlit_app/app.py

# Start all services (MinIO, Prometheus, Grafana, Ray, and Streamlit)
start:
	scripts/init.sh

# Stop all services
stop:
	scripts/stop.sh

# Setup Ray metrics and import Grafana dashboards
setup-metrics:
	scripts/setup-metrics.sh

# Start KubeRay cluster with all services
kuberay-start:
	scripts/kuberay-init.sh

# Stop KubeRay cluster and all services
kuberay-stop:
	scripts/kuberay-stop.sh

# Check KubeRay cluster status
kuberay-status:
	scripts/kuberay-status.sh
