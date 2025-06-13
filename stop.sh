#!/bin/bash
# ML Homelab Shutdown Script
# This script stops all services started by init.sh

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${RED}${BOLD}=== ML Homelab Shutdown ===${NC}"
echo -e "Stopping all services...\n"

# Step 1: Stop Streamlit
echo -e "${YELLOW}Step 1/3: Stopping Streamlit...${NC}"
STREAMLIT_PID=$(ps aux | grep "streamlit run streamlit_app/app.py" | grep -v grep | awk '{print $2}')
if [ -n "$STREAMLIT_PID" ]; then
  echo -e "Killing Streamlit process (PID: $STREAMLIT_PID)..."
  kill $STREAMLIT_PID
  echo -e "✅ Streamlit stopped\n"
else
  echo -e "No Streamlit process found to stop\n"
fi

# Step 2: Stop Ray
echo -e "${YELLOW}Step 2/3: Stopping Ray cluster...${NC}"
ray stop

# Step 3: Stop MinIO
echo -e "${YELLOW}Step 3/3: Stopping MinIO with Docker Compose...${NC}"
if docker ps | grep -q "minio"; then
  echo -e "Shutting down MinIO containers..."
  docker compose down
  echo -e "✅ MinIO stopped\n"
else
  echo -e "No MinIO containers found to stop\n"
fi

# Success message
echo -e "${GREEN}${BOLD}=== ML Homelab shutdown complete! ===${NC}"
echo -e "All services have been stopped.\n"
