#!/bin/bash
# KubeRay Initialization Script
# This script sets up a KubeRay cluster locally using Kind and Helm

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

# Function to detect architecture
detect_architecture() {
  local arch=$(uname -m)
  case $arch in
    arm64|aarch64)
      echo "aarch64"
      ;;
    x86_64|amd64)
      echo "x86_64"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

echo -e "${BLUE}${BOLD}=== KubeRay Cluster Initialization ===${NC}"
echo -e "Starting KubeRay cluster setup...\n"

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1/7: Checking prerequisites...${NC}"
missing_deps=0

if ! command_exists kind; then
  echo -e "❌ Kind not found. Installing with brew..."
  if command_exists brew; then
    brew install kind
  else
    echo -e "❌ Homebrew not found. Please install Kind manually: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    missing_deps=1
  fi
fi

if ! command_exists helm; then
  echo -e "❌ Helm not found. Installing with brew..."
  if command_exists brew; then
    brew install helm
  else
    echo -e "❌ Homebrew not found. Please install Helm manually: https://helm.sh/docs/intro/install/"
    missing_deps=1
  fi
fi

if ! command_exists kubectl; then
  echo -e "❌ kubectl not found. Installing with brew..."
  if command_exists brew; then
    brew install kubectl
  else
    echo -e "❌ Homebrew not found. Please install kubectl manually: https://kubernetes.io/docs/tasks/tools/"
    missing_deps=1
  fi
fi

if ! command_exists docker; then
  echo -e "❌ Docker not found. Please install Docker from https://www.docker.com/"
  missing_deps=1
fi

if ! command_exists streamlit; then
  echo -e "❌ Streamlit not found. Please install it using: pip install streamlit"
  missing_deps=1
fi

if [ $missing_deps -eq 1 ]; then
  echo -e "\n${RED}Please install the missing dependencies and try again.${NC}"
  exit 1
fi

echo -e "✅ All required prerequisites are installed\n"

# Step 2: Create Kind cluster
echo -e "${YELLOW}Step 2/7: Creating Kind cluster...${NC}"
if kind get clusters | grep -q "kind"; then
  echo -e "Kind cluster already exists"
else
  echo -e "Creating Kind cluster with Kubernetes v1.33.1..."
  kind create cluster --image=kindest/node:v1.33.1
  if [ $? -eq 0 ]; then
    echo -e "✅ Kind cluster created successfully"
  else
    echo -e "❌ Failed to create Kind cluster"
    exit 1
  fi
fi
echo

# Step 3: Install KubeRay operator
echo -e "${YELLOW}Step 3/7: Installing KubeRay operator...${NC}"
echo -e "Adding KubeRay Helm repository..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

echo -e "Checking if KubeRay operator is already installed..."
if helm list -n kuberay-system | grep -q "kuberay-operator"; then
  echo -e "KubeRay operator already exists"
else
  echo -e "Installing KubeRay operator..."
  helm install kuberay-operator kuberay/kuberay-operator --create-namespace --namespace kuberay-system
  if [ $? -eq 0 ]; then
    echo -e "✅ KubeRay operator installed successfully"
  else
    echo -e "❌ Failed to install KubeRay operator"
    exit 1
  fi
fi
echo

# Step 4: Install Ray cluster
echo -e "${YELLOW}Step 4/7: Installing Ray cluster...${NC}"
ARCH=$(detect_architecture)
echo -e "Detected architecture: ${ARCH}"

echo -e "Checking if Ray cluster is already installed..."
if helm list | grep -q "raycluster"; then
  echo -e "Ray cluster already exists"
else
  if [ "$ARCH" = "aarch64" ]; then
    echo -e "Installing Ray cluster for ARM architecture..."
    helm install raycluster kuberay/ray-cluster --version 1.3.2 \
      --set 'image.tag=2.47.0-py311-aarch64' \
      --set 'head.rayStartParams.port=6379' \
      --set 'head.rayStartParams.dashboard-port=8265' \
      --set 'head.rayStartParams.ray-client-server-port=10001' \
      --set 'head.rayStartParams.node-ip-address=0.0.0.0' \
      --set 'head.rayStartParams.dashboard-host=0.0.0.0' \
      --set 'worker.rayStartParams.node-ip-address=0.0.0.0'
  else
    echo -e "Installing Ray cluster for x86_64 architecture..."
    helm install raycluster kuberay/ray-cluster --version 1.3.2 \
      --set 'image.tag=2.47.0-py311' \
      --set 'head.rayStartParams.port=6379' \
      --set 'head.rayStartParams.dashboard-port=8265' \
      --set 'head.rayStartParams.ray-client-server-port=10001' \
      --set 'head.rayStartParams.node-ip-address=0.0.0.0' \
      --set 'head.rayStartParams.dashboard-host=0.0.0.0' \
      --set 'worker.rayStartParams.node-ip-address=0.0.0.0'
  fi

  if [ $? -eq 0 ]; then
    echo -e "✅ Ray cluster installed successfully"
  else
    echo -e "❌ Failed to install Ray cluster"
    exit 1
  fi
fi
echo

# Step 5: Wait for cluster to be ready
echo -e "${YELLOW}Step 5/7: Waiting for Ray cluster to be ready...${NC}"
echo -e "⏳ Waiting for Ray cluster pods to be created..."

# Wait for pods to be created (not necessarily ready)
for i in {1..30}; do
  pod_count=$(kubectl get pods --selector=ray.io/cluster=raycluster-kuberay --no-headers 2>/dev/null | wc -l)
  if [ "$pod_count" -gt 0 ]; then
    echo -e "✅ Ray cluster pods created (found $pod_count pods)"
    break
  elif [ $i -eq 30 ]; then
    echo -e "❌ Ray cluster pods failed to be created within timeout"
    exit 1
  else
    echo -e "⏳ Attempt $i: Waiting for pods to be created..."
    sleep 5
  fi
done

# Wait for Ray head service to be fully ready (most important)
echo -e "⏳ Waiting for Ray head service to be available..."
kubectl wait --for=condition=ready pod --selector=ray.io/cluster=raycluster-kuberay,ray.io/node-type=head --timeout=300s

if [ $? -eq 0 ]; then
  echo -e "✅ Ray head service is ready"
else
  echo -e "❌ Ray head service failed to start"
  exit 1
fi

# Check worker pods status (non-blocking)
echo -e "⏳ Checking worker pod status..."
worker_ready=$(kubectl get pods --selector=ray.io/cluster=raycluster-kuberay,ray.io/node-type=worker --no-headers 2>/dev/null | grep "1/1.*Running" | wc -l)
worker_total=$(kubectl get pods --selector=ray.io/cluster=raycluster-kuberay,ray.io/node-type=worker --no-headers 2>/dev/null | wc -l)

if [ "$worker_ready" -gt 0 ]; then
  echo -e "✅ Worker pods ready: $worker_ready/$worker_total"
else
  echo -e "⚠️ Worker pods not ready: $worker_ready/$worker_total (Ray cluster will work with head node only)"
fi

# Additional wait to ensure Ray dashboard is fully initialized
echo -e "⏳ Waiting for Ray dashboard to initialize..."
sleep 20
echo

# Step 6: Start MinIO with Docker Compose
echo -e "${YELLOW}Step 6/7: Starting MinIO with Docker Compose...${NC}"
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
  echo -e "✅ MinIO started successfully"
else
  echo -e "⚠️ MinIO may not have started properly, but continuing anyway..."
fi
echo

# Step 7: Setup port forwarding and start Streamlit
echo -e "${YELLOW}Step 7/7: Setting up port forwarding and starting services...${NC}"

# Clean up any existing port forwarding
pkill -f "kubectl port-forward.*8265" 2>/dev/null || true
pkill -f "kubectl port-forward.*6379" 2>/dev/null || true
pkill -f "kubectl port-forward.*10001" 2>/dev/null || true
sleep 2

# Port forward Ray dashboard, GCS server, and client server
echo -e "Setting up port forwarding for Ray dashboard (8265), GCS server (6379), and client server (10001)..."
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 6379:6379 10001:10001 > /dev/null 2>&1 &
RAY_PORT_FORWARD_PID=$!
echo -e "⏳ Waiting for port forwarding to establish..."
sleep 5

# Verify Ray dashboard accessibility
echo -e "Verifying Ray dashboard accessibility..."
for i in {1..10}; do
  if curl -s -o /dev/null http://localhost:8265; then
    echo -e "✅ Ray dashboard is accessible on attempt $i"
    break
  elif [ $i -eq 10 ]; then
    echo -e "❌ Ray dashboard not accessible after 10 attempts"
    echo -e "Troubleshooting steps:"
    echo -e "1. Check if port forwarding is running: ps aux | grep 'kubectl port-forward'"
    echo -e "2. Check Ray head pod logs: kubectl logs -l ray.io/node-type=head"
    echo -e "3. Check service: kubectl get svc raycluster-kuberay-head-svc"
    exit 1
  else
    echo -e "⏳ Attempt $i: Ray dashboard not ready, retrying in 3 seconds..."
    sleep 3
  fi
done

# Verify GCS server accessibility
echo -e "Verifying Ray GCS server accessibility..."
for i in {1..5}; do
  if nc -z localhost 6379 2>/dev/null; then
    echo -e "✅ Ray GCS server is accessible on attempt $i"
    break
  elif [ $i -eq 5 ]; then
    echo -e "⚠️ Ray GCS server may not be accessible, but continuing..."
  else
    echo -e "⏳ Attempt $i: Ray GCS server not ready, retrying in 2 seconds..."
    sleep 2
  fi
done

# Start Streamlit app
echo -e "Starting Streamlit app..."
if is_port_in_use 8501; then
  echo -e "Streamlit seems to be already running on port 8501"
else
  echo -e "Starting Streamlit app with 'make run'..."
  (cd "$(dirname "$0")/.." && make run) &
  STREAMLIT_PID=$!
  echo -e "⏳ Waiting for Streamlit to initialize..."
  sleep 5
fi

# Check if Streamlit is responsive
curl -s -o /dev/null http://localhost:8501
if [ $? -eq 0 ]; then
  echo -e "✅ Streamlit started successfully"
else
  echo -e "⚠️ Streamlit may not have started properly, but continuing anyway..."
fi

echo

# Success message
echo -e "${GREEN}${BOLD}=== KubeRay Cluster started successfully! ===${NC}"
echo -e "All services are now running. You can access:"
echo -e "- ${BLUE}MinIO Console:${NC} http://localhost:9001/ (credentials from .env)"
echo -e "- ${BLUE}Streamlit Dashboard:${NC} http://localhost:8501/"
echo -e "- ${BLUE}Ray Dashboard:${NC} http://localhost:8265/"
echo
echo -e "${YELLOW}KubeRay Cluster Status:${NC}"
kubectl get rayclusters
echo
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay
echo
kubectl get service raycluster-kuberay-head-svc
echo
echo -e "${YELLOW}To submit a job to the Ray cluster:${NC}"
echo -e "ray job submit --address http://localhost:8265 -- python -c \"import pprint; import ray; ray.init(); pprint.pprint(ray.cluster_resources(), sort_dicts=True)\""
echo
echo -e "${YELLOW}Note:${NC} To stop all services, run: ./kuberay-stop.sh"
echo -e "${YELLOW}Note:${NC} To check cluster status, run: ./kuberay-status.sh"
echo -e "${YELLOW}Note:${NC} Port forwarding PID: $RAY_PORT_FORWARD_PID"

# Store PIDs for cleanup
echo $RAY_PORT_FORWARD_PID > /tmp/kuberay-port-forward.pid
if [ ! -z "$STREAMLIT_PID" ]; then
  echo $STREAMLIT_PID > /tmp/kuberay-streamlit.pid
fi

# Wait for Ctrl+C
if [ ! -z "$STREAMLIT_PID" ]; then
  wait $STREAMLIT_PID
fi