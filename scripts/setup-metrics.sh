#!/bin/bash
# Setup script for Ray metrics with Grafana dashboards
# This script copies Ray's default Grafana dashboards to the provisioning directory

set -e

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}=== Ray Metrics Setup ===${NC}\n"

# Check if Ray is running
if ! pgrep -x "ray" > /dev/null; then
    echo -e "${YELLOW}Ray is not running. Starting Ray with metrics enabled...${NC}"
    ray start --head \
        --port=6379 \
        --dashboard-port=8265 \
        --ray-client-server-port=10001 \
        --metrics-export-port=8080 \
        --node-ip-address=127.0.0.1 \
        --dashboard-host=127.0.0.1
    
    echo -e "⏳ Waiting for Ray to initialize..."
    sleep 5
    echo -e "${GREEN}✓ Ray started${NC}\n"
else
    echo -e "${GREEN}✓ Ray is already running${NC}\n"
fi

# Check if Ray session directory exists
RAY_DASHBOARD_DIR="/tmp/ray/session_latest/metrics/grafana/dashboards"
if [ ! -d "$RAY_DASHBOARD_DIR" ]; then
    echo -e "${RED}✗ Ray dashboard directory not found: $RAY_DASHBOARD_DIR${NC}"
    echo -e "${YELLOW}Make sure Ray is running with metrics enabled.${NC}"
    exit 1
fi

# Get the script directory (ml-homelab root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DEST="$PROJECT_ROOT/config/grafana/provisioning/dashboards/json"

# Create destination directory if it doesn't exist
mkdir -p "$DASHBOARD_DEST"

# Copy dashboard JSON files
echo -e "${YELLOW}Copying Ray default dashboards...${NC}"
COPIED_COUNT=0

if ls "$RAY_DASHBOARD_DIR"/*.json 1> /dev/null 2>&1; then
    for dashboard in "$RAY_DASHBOARD_DIR"/*.json; do
        dashboard_name=$(basename "$dashboard")
        cp "$dashboard" "$DASHBOARD_DEST/"
        echo -e "  ${GREEN}✓${NC} Copied: $dashboard_name"
        ((COPIED_COUNT++))
    done
    echo -e "\n${GREEN}Successfully copied $COPIED_COUNT dashboard(s)${NC}\n"
else
    echo -e "${YELLOW}No dashboard JSON files found in $RAY_DASHBOARD_DIR${NC}"
    echo -e "${YELLOW}Ray may not have generated dashboards yet.${NC}\n"
fi

# Check if Grafana is running
echo -e "${YELLOW}Checking Grafana status...${NC}"
if docker compose ps grafana | grep -q "Up"; then
    echo -e "${GREEN}✓ Grafana is running${NC}"
    echo -e "${YELLOW}Restarting Grafana to load new dashboards...${NC}"
    
    cd "$PROJECT_ROOT"
    docker compose restart grafana
    
    echo -e "⏳ Waiting for Grafana to restart..."
    sleep 5
    echo -e "${GREEN}✓ Grafana restarted${NC}\n"
else
    echo -e "${YELLOW}⚠ Grafana is not running${NC}"
    echo -e "${YELLOW}Start Grafana with: docker compose up -d grafana${NC}\n"
fi

# Verify metrics endpoint
echo -e "${YELLOW}Verifying Ray metrics endpoint...${NC}"
if curl -s -f http://localhost:8080/metrics > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ray metrics endpoint is accessible at http://localhost:8080/metrics${NC}\n"
else
    echo -e "${RED}✗ Ray metrics endpoint is not accessible${NC}"
    echo -e "${YELLOW}Make sure Ray is started with --metrics-export-port=8080${NC}\n"
fi

# Success message
echo -e "${GREEN}${BOLD}=== Setup Complete! ===${NC}\n"
echo -e "You can now access:"
echo -e "- ${BLUE}Grafana:${NC} http://localhost:3000/ (admin/admin)"
echo -e "- ${BLUE}Prometheus:${NC} http://localhost:9090/"
echo -e "- ${BLUE}Ray Dashboard:${NC} http://localhost:8265/"
echo -e "\nTo view the imported dashboards:"
echo -e "1. Open Grafana at http://localhost:3000/"
echo -e "2. Navigate to: Dashboards → Browse → Ray"
echo -e "\n${YELLOW}Note:${NC} If dashboards don't appear, check Grafana logs with:"
echo -e "  docker compose logs grafana\n"


