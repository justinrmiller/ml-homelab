.PHONY: run start stop

# Start Streamlit app only
run:
	streamlit run streamlit_app/app.py

# Start all services (MinIO, Ray, and Streamlit)
start:
	./init.sh

# Stop all services
stop:
	./stop.sh