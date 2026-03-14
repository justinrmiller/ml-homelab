#!/bin/bash
# KubeRay Status Script
# This script checks the status of all KubeRay services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Function to check service health
check_service_health() {
  local url=$1
  local service_name=$2

  if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|404"; then
    echo -e "✅ $service_name is ${GREEN}healthy${NC}"
  else
    echo -e "❌ $service_name is ${RED}unhealthy${NC}"
  fi
}

echo -e "${BLUE}${BOLD}=== KubeRay Cluster Status ===${NC}"
echo -e "Checking status of all services...\n"

# Detect container runtime early (needed for kind_cmd and compose checks)
detect_container_runtime
echo

# Check if required tools are available
echo -e "${YELLOW}Prerequisites Check:${NC}"
if command_exists uv; then
  echo -e "✅ uv: $(uv --version)"
else
  echo -e "❌ uv: Not available"
fi

if command_exists kubectl; then
  echo -e "✅ kubectl: Available"
else
  echo -e "❌ kubectl: Not available"
fi

if command_exists helm; then
  echo -e "✅ helm: Available"
else
  echo -e "❌ helm: Not available"
fi

if command_exists kind; then
  echo -e "✅ kind: Available"
else
  echo -e "❌ kind: Not available"
fi

if command_exists docker; then
  echo -e "✅ docker: Available"
elif command_exists podman; then
  echo -e "✅ podman: Available"
else
  echo -e "❌ docker/podman: Not available"
fi
echo

# Check Kind cluster status
echo -e "${YELLOW}Kind Cluster Status:${NC}"
if command_exists kind; then
  if kind_cmd get clusters | grep -q "kind"; then
    echo -e "✅ Kind cluster: ${GREEN}Running${NC}"

    # Check cluster info
    echo -e "${BLUE}Cluster Info:${NC}"
    kubectl cluster-info --context kind-kind 2>/dev/null || echo -e "❌ Cannot connect to cluster"
  else
    echo -e "❌ Kind cluster: ${RED}Not running${NC}"
  fi
else
  echo -e "❌ Kind not available"
fi
echo

# Check KubeRay operator status
echo -e "${YELLOW}KubeRay Operator Status:${NC}"
if command_exists helm && command_exists kubectl; then
  if helm list -n kuberay-system | grep -q "kuberay-operator"; then
    echo -e "✅ KubeRay operator: ${GREEN}Installed${NC}"

    # Check operator pods
    echo -e "${BLUE}Operator Pods:${NC}"
    kubectl get pods -n kuberay-system --selector=app.kubernetes.io/name=kuberay-operator 2>/dev/null || echo -e "❌ Cannot get operator pods"
  else
    echo -e "❌ KubeRay operator: ${RED}Not installed${NC}"
  fi
else
  echo -e "❌ Required tools not available"
fi
echo

# Check Ray cluster status
echo -e "${YELLOW}Ray Cluster Status:${NC}"
if command_exists helm && command_exists kubectl; then
  if helm list | grep -q "raycluster"; then
    echo -e "✅ Ray cluster: ${GREEN}Installed${NC}"

    # Check Ray cluster resource
    echo -e "${BLUE}Ray Clusters:${NC}"
    kubectl get rayclusters 2>/dev/null || echo -e "❌ Cannot get Ray clusters"

    echo -e "${BLUE}Ray Cluster Pods:${NC}"
    kubectl get pods --selector=ray.io/cluster=raycluster-kuberay 2>/dev/null || echo -e "❌ Cannot get Ray cluster pods"

    echo -e "${BLUE}Ray Head Service:${NC}"
    kubectl get service raycluster-kuberay-head-svc 2>/dev/null || echo -e "❌ Cannot get Ray head service"
  else
    echo -e "❌ Ray cluster: ${RED}Not installed${NC}"
  fi
else
  echo -e "❌ Required tools not available"
fi
echo

# Check port forwarding status
echo -e "${YELLOW}Port Forwarding Status:${NC}"
RAY_PORT_FORWARD=$(ps aux | grep "kubectl port-forward.*raycluster-kuberay-head-svc.*8265" | grep -v grep)
if [ -n "$RAY_PORT_FORWARD" ]; then
  echo -e "✅ Ray dashboard port forward: ${GREEN}Active${NC}"
  echo -e "   Process: $RAY_PORT_FORWARD"
else
  echo -e "❌ Ray dashboard port forward: ${RED}Not active${NC}"
fi
echo

# Use previously detected container runtime
_CRT="$CONTAINER_RT"

# Check Compose services status
echo -e "${YELLOW}Container Services:${NC}"
if [ -n "$_CRT" ]; then
  if $_CRT ps --format "table {{.Names}}\t{{.Status}}" | grep -q "minio"; then
    echo -e "✅ MinIO containers: ${GREEN}Running${NC}"
    $_CRT ps --format "table {{.Names}}\t{{.Status}}" | grep minio
  else
    echo -e "❌ MinIO containers: ${RED}Not running${NC}"
  fi

  if $_CRT ps --format "table {{.Names}}\t{{.Status}}" | grep -q "prometheus"; then
    echo -e "✅ Prometheus: ${GREEN}Running${NC}"
  else
    echo -e "❌ Prometheus: ${RED}Not running${NC}"
  fi

  if $_CRT ps --format "table {{.Names}}\t{{.Status}}" | grep -q "grafana"; then
    echo -e "✅ Grafana: ${GREEN}Running${NC}"
  else
    echo -e "❌ Grafana: ${RED}Not running${NC}"
  fi
else
  echo -e "❌ No container runtime available"
fi
echo

# Check service accessibility
echo -e "${YELLOW}Service Accessibility:${NC}"

# Check Ray dashboard
if is_port_in_use 8265; then
  echo -e "✅ Ray dashboard (8265): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:8265" "Ray dashboard"
else
  echo -e "❌ Ray dashboard (8265): ${RED}Port not available${NC}"
fi

# Check MinIO API
if is_port_in_use 9000; then
  echo -e "✅ MinIO API (9000): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:9000" "MinIO API"
else
  echo -e "❌ MinIO API (9000): ${RED}Port not available${NC}"
fi

# Check MinIO Console
if is_port_in_use 9001; then
  echo -e "✅ MinIO Console (9001): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:9001" "MinIO Console"
else
  echo -e "❌ MinIO Console (9001): ${RED}Port not available${NC}"
fi

# Check Prometheus
if is_port_in_use 9090; then
  echo -e "✅ Prometheus (9090): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:9090" "Prometheus"
else
  echo -e "❌ Prometheus (9090): ${RED}Port not available${NC}"
fi

# Check Grafana
if is_port_in_use 3000; then
  echo -e "✅ Grafana (3000): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:3000" "Grafana"
else
  echo -e "❌ Grafana (3000): ${RED}Port not available${NC}"
fi

# Check Streamlit
if is_port_in_use 8501; then
  echo -e "✅ Streamlit (8501): ${GREEN}Port available${NC}"
  check_service_health "http://localhost:8501" "Streamlit"
else
  echo -e "❌ Streamlit (8501): ${RED}Port not available${NC}"
fi
echo

# Check Streamlit process
echo -e "${YELLOW}Streamlit Process Status:${NC}"
STREAMLIT_PID=$(ps aux | grep "streamlit run streamlit_app/app.py" | grep -v grep | awk '{print $2}')
if [ -n "$STREAMLIT_PID" ]; then
  echo -e "✅ Streamlit process: ${GREEN}Running${NC} (PID: $STREAMLIT_PID)"
else
  echo -e "❌ Streamlit process: ${RED}Not running${NC}"
fi
echo

# Summary
echo -e "${BLUE}${BOLD}=== Status Summary ===${NC}"
echo -e "Access URLs:"
echo -e "- ${BLUE}Ray Dashboard:${NC} http://localhost:8265/"
echo -e "- ${BLUE}MinIO Console:${NC} http://localhost:9001/"
echo -e "- ${BLUE}Prometheus:${NC} http://localhost:9090/"
echo -e "- ${BLUE}Grafana:${NC} http://localhost:3000/ (admin/admin)"
echo -e "- ${BLUE}Streamlit App:${NC} http://localhost:8501/"
echo
echo -e "Management Commands:"
echo -e "- ${BLUE}Submit Ray job:${NC} uv run ray job submit --address http://localhost:8265 -- python -c \"import ray; ray.init(); print(ray.cluster_resources())\""
echo -e "- ${BLUE}Stop cluster:${NC} make stop"
echo -e "- ${BLUE}Restart cluster:${NC} make stop && make start"
