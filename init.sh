#!/bin/bash
# ML Homelab Initialization Script
# This script starts all required services for the ML Homelab project

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running on a specific port
is_port_in_use() {
  netstat -an | grep -q "LISTEN" | grep -q "$1"
  return $?
}

echo -e "${BLUE}${BOLD}=== ML Homelab Initialization ===${NC}"
echo -e "Starting all required services...\n"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
missing_deps=0

if ! command_exists docker; then
  echo -e "❌ Docker not found. Please install Docker from https://www.docker.com/"
  missing_deps=1
fi

if ! command_exists ray; then
  echo -e "❌ Ray not found. Please install Ray using: pip install ray[default]==2.46.0"
  missing_deps=1
fi

if ! command_exists streamlit; then
  echo -e "❌ Streamlit not found. Please install it using: pip install streamlit"
  missing_deps=1
fi

if [ $missing_deps -eq 1 ]; then
  echo -e "\n${YELLOW}Please install the missing dependencies and try again.${NC}"
  exit 1
fi

echo -e "✅ All required prerequisites are installed\n"

# Step 1: Start MinIO using Docker Compose
echo -e "${YELLOW}Step 1/3: Starting MinIO with Docker Compose...${NC}"
if is_port_in_use 9000; then
  echo -e "MinIO seems to be already running on port 9000"
else
  docker compose up -d
  echo -e "⏳ Waiting for MinIO to initialize..."
  sleep 5
fi

# Check if MinIO is responsive
curl -s -o /dev/null http://localhost:9000
if [ $? -eq 0 ]; then
  echo -e "✅ MinIO started successfully\n"
else
  echo -e "⚠️ MinIO may not have started properly, but continuing anyway...\n"
fi

# Step 2: Start Ray
echo -e "${YELLOW}Step 2/3: Starting Ray cluster...${NC}"
if is_port_in_use 8265; then
  echo -e "Ray seems to be already running on port 8265"
else
  ray start --head --port=6379 --dashboard-port=8265 &
  RAY_PID=$!
  echo -e "⏳ Waiting for Ray to initialize..."
  sleep 5
fi

# Check if Ray dashboard is responsive
curl -s -o /dev/null http://localhost:8265
if [ $? -eq 0 ]; then
  echo -e "✅ Ray started successfully\n"
else
  echo -e "⚠️ Ray may not have started properly, but continuing anyway...\n"
fi

# Step 3: Start Streamlit app
echo -e "${YELLOW}Step 3/3: Starting Streamlit app...${NC}"
if is_port_in_use 8501; then
  echo -e "Streamlit seems to be already running on port 8501"
else
  echo -e "Starting Streamlit app with 'make run'..."
  (cd "$(dirname "$0")" && make run) &
  STREAMLIT_PID=$!
  echo -e "⏳ Waiting for Streamlit to initialize..."
  sleep 5
fi

# Check if Streamlit is responsive
curl -s -o /dev/null http://localhost:8501
if [ $? -eq 0 ]; then
  echo -e "✅ Streamlit started successfully\n"
else
  echo -e "⚠️ Streamlit may not have started properly, but continuing anyway...\n"
fi

# Success message
echo -e "${GREEN}${BOLD}=== ML Homelab started successfully! ===${NC}"
echo -e "All services are now running. You can access:"
echo -e "- ${BLUE}MinIO Console:${NC} http://localhost:9001/ (credentials from .env)"
echo -e "- ${BLUE}Streamlit Dashboard:${NC} http://localhost:8501/"
echo -e "- ${BLUE}Ray Dashboard:${NC} http://localhost:8265/"
echo -e "\n${YELLOW}Note:${NC} To stop all services, run: ./stop.sh"
echo -e "${YELLOW}Note:${NC} Press Ctrl+C to stop the script, but services will continue running in the background\n"

# Wait for Ctrl+C
wait $STREAMLIT_PID
