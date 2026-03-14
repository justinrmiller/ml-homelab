.PHONY: run start stop status sync job

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

# Submit a Ray job (usage: make job SCRIPT=hello_ray_job.py)
job:
	uv run ray job submit --address http://localhost:8265 -- python $(SCRIPT)
