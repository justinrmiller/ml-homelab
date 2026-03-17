#!/bin/bash
# KubeRay Shutdown Script
# This script stops all KubeRay services and cleans up the cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo -e "${RED}${BOLD}=== KubeRay Cluster Shutdown ===${NC}"
echo -e "Stopping all KubeRay services...\n"

# Step 1: Stop Streamlit
echo -e "${YELLOW}Step 1/6: Stopping Streamlit...${NC}"
STREAMLIT_PID=$(ps aux | grep "streamlit run streamlit_app/app.py" | grep -v grep | awk '{print $2}')
if [ -n "$STREAMLIT_PID" ]; then
  echo -e "Killing Streamlit process (PID: $STREAMLIT_PID)..."
  kill $STREAMLIT_PID
  echo -e "✅ Streamlit stopped"
else
  echo -e "No Streamlit process found to stop"
fi

# Also check for PID file
if [ -f /tmp/kuberay-streamlit.pid ]; then
  STREAMLIT_PID=$(cat /tmp/kuberay-streamlit.pid)
  if ps -p $STREAMLIT_PID > /dev/null 2>&1; then
    echo -e "Killing Streamlit process from PID file (PID: $STREAMLIT_PID)..."
    kill $STREAMLIT_PID
  fi
  rm -f /tmp/kuberay-streamlit.pid
fi
echo

# Step 2: Stop port forwarding
echo -e "${YELLOW}Step 2/6: Stopping port forwarding...${NC}"
# Kill the auto-reconnecting port-forward loop and all kubectl port-forward processes
pkill -f "kuberay-port-forward-loop" 2>/dev/null || true
rm -f /tmp/kuberay-port-forward-loop.sh
KUBECTL_PIDS=$(ps aux | grep "kubectl port-forward" | grep -v grep | awk '{print $2}')
if [ -n "$KUBECTL_PIDS" ]; then
  echo -e "Killing kubectl port-forward processes..."
  echo $KUBECTL_PIDS | xargs kill
  echo -e "✅ Port forwarding stopped"
else
  echo -e "No kubectl port-forward processes found"
fi

# Also check for PID file
if [ -f /tmp/kuberay-port-forward.pid ]; then
  PORT_FORWARD_PID=$(cat /tmp/kuberay-port-forward.pid)
  if ps -p $PORT_FORWARD_PID > /dev/null 2>&1; then
    echo -e "Killing port-forward process from PID file (PID: $PORT_FORWARD_PID)..."
    kill $PORT_FORWARD_PID
  fi
  rm -f /tmp/kuberay-port-forward.pid
fi
echo

# Step 3: Stop Compose services
echo -e "${YELLOW}Step 3/6: Stopping Compose services...${NC}"
detect_container_runtime
if $CONTAINER_RT ps | grep -q "minio"; then
  echo -e "Shutting down containers..."
  $COMPOSE_CMD down
  echo -e "✅ Compose services stopped"
else
  echo -e "No containers found to stop"
fi
echo

# Step 4: Uninstall Ray cluster
echo -e "${YELLOW}Step 4/6: Uninstalling Ray cluster...${NC}"
if helm list | grep -q "raycluster"; then
  echo -e "Uninstalling Ray cluster..."
  helm uninstall raycluster
  echo -e "✅ Ray cluster uninstalled"
else
  echo -e "No Ray cluster found to uninstall"
fi
echo

# Step 5: Uninstall KubeRay operator
echo -e "${YELLOW}Step 5/6: Uninstalling KubeRay operator...${NC}"
if helm list -n kuberay-system | grep -q "kuberay-operator"; then
  echo -e "Uninstalling KubeRay operator..."
  helm uninstall kuberay-operator -n kuberay-system
  echo -e "✅ KubeRay operator uninstalled"
else
  echo -e "No KubeRay operator found to uninstall"
fi
echo

# Step 6: Delete Kind cluster
echo -e "${YELLOW}Step 6/6: Deleting Kind cluster...${NC}"
if kind_cmd get clusters | grep -q "kind"; then
  echo -e "Deleting Kind cluster..."
  kind_cmd delete cluster
  echo -e "✅ Kind cluster deleted"
else
  echo -e "No Kind cluster found to delete"
fi
echo

# Clean up any remaining processes
echo -e "${YELLOW}Cleaning up remaining processes...${NC}"
# Kill any remaining kubectl processes
killall kubectl > /dev/null 2>&1 || true
echo -e "✅ Process cleanup completed"
echo

# Success message
echo -e "${GREEN}${BOLD}=== KubeRay Cluster shutdown complete! ===${NC}"
echo -e "All services have been stopped and cleaned up."
echo -e "- Kind cluster deleted"
echo -e "- Helm releases uninstalled"
echo -e "- Port forwarding stopped"
echo -e "- MinIO containers stopped"
echo -e "- Streamlit process terminated"
echo