#!/bin/bash
# ML Homelab Initialization Script
# This script starts all required services for the ML Homelab project

# Parse command line arguments
USE_KUBERAY=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --kuberay)
      USE_KUBERAY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --kuberay    Use KubeRay instead of local Ray cluster"
      echo "  -h, --help   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

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

# Check if KubeRay mode is requested
if [ "$USE_KUBERAY" = true ]; then
  echo -e "${BLUE}${BOLD}=== ML Homelab Initialization (KubeRay Mode) ===${NC}"
  echo -e "Delegating to KubeRay initialization script...\n"
  exec "$(dirname "$0")/kuberay-init.sh"
  exit $?
fi

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
  echo -e "❌ Ray not found. Please install Ray using: pip install ray[default]==2.49.2"
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

# Step 1: Start MinIO, Prometheus, and Grafana using Docker Compose
echo -e "${YELLOW}Step 1/4: Starting services with Docker Compose (MinIO, Prometheus, Grafana)...${NC}"
if is_port_in_use 9000; then
  echo -e "MinIO seems to be already running on port 9000"
else
  docker compose up -d
  echo -e "⏳ Waiting for MinIO to initialize..."
  sleep 5
fi

# Check if services are responsive
curl -s -o /dev/null http://localhost:9000
if [ $? -eq 0 ]; then
  echo -e "✅ MinIO started successfully"
fi

curl -s -o /dev/null http://localhost:9090
if [ $? -eq 0 ]; then
  echo -e "✅ Prometheus started successfully"
fi

curl -s -o /dev/null http://localhost:3000
if [ $? -eq 0 ]; then
  echo -e "✅ Grafana started successfully"
fi
echo ""

# Step 2: Start Ray
echo -e "${YELLOW}Step 2/4: Starting Ray cluster with metrics enabled...${NC}"
if is_port_in_use 8265; then
  echo -e "Ray seems to be already running on port 8265"
else
  ray start --head \
    --port=6379 \
    --dashboard-port=8265 \
    --ray-client-server-port=10001 \
    --metrics-export-port=8080 \
    --node-ip-address=127.0.0.1 \
    --dashboard-host=127.0.0.1 &
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

# Step 2.5: Auto-setup Grafana dashboards from Ray
echo -e "${YELLOW}Setting up Grafana dashboards from Ray...${NC}"
RAY_DASHBOARD_DIR="/tmp/ray/session_latest/metrics/grafana/dashboards"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DEST="$PROJECT_ROOT/config/grafana/provisioning/dashboards/json"

# Create destination directory if it doesn't exist
mkdir -p "$DASHBOARD_DEST"

# Wait a moment for Ray to generate dashboard files
sleep 2

# Copy dashboard JSON files if they exist
if [ -d "$RAY_DASHBOARD_DIR" ] && ls "$RAY_DASHBOARD_DIR"/*.json 1> /dev/null 2>&1; then
  COPIED_COUNT=0
  for dashboard in "$RAY_DASHBOARD_DIR"/*.json; do
    dashboard_name=$(basename "$dashboard")
    cp "$dashboard" "$DASHBOARD_DEST/" 2>/dev/null
    ((COPIED_COUNT++))
  done
  echo -e "✅ Copied $COPIED_COUNT Ray dashboard(s) to Grafana"
  
  # Restart Grafana to load new dashboards if it's running
  if docker compose ps grafana 2>/dev/null | grep -q "Up"; then
    echo -e "⏳ Restarting Grafana to load dashboards..."
    (cd "$PROJECT_ROOT" && docker compose restart grafana > /dev/null 2>&1)
    sleep 3
    echo -e "✅ Grafana restarted with new dashboards\n"
  else
    echo -e "${YELLOW}⚠ Grafana not running yet. Dashboards will load when Grafana starts.\n${NC}"
  fi
else
  echo -e "${YELLOW}⚠ Ray dashboards not found yet. You can run 'make setup-metrics' later to import them.\n${NC}"
fi

# Step 3: Start Streamlit app
echo -e "${YELLOW}Step 3/4: Starting Streamlit app...${NC}"
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
echo -e "- ${BLUE}Prometheus:${NC} http://localhost:9090/"
echo -e "- ${BLUE}Grafana:${NC} http://localhost:3000/ (admin/admin)"
echo -e "\n${YELLOW}Metrics:${NC} Ray metrics are exported on port 8080 and scraped by Prometheus"
echo -e "${YELLOW}Grafana:${NC} Ray dashboards have been automatically imported and are available in Grafana"
echo -e "${YELLOW}Note:${NC} To stop all services, run: ./stop.sh"
echo -e "${YELLOW}Note:${NC} Press Ctrl+C to stop the script, but services will continue running in the background\n"

# Wait for Ctrl+C
wait $STREAMLIT_PID
