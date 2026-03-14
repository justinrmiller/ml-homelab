#!/bin/bash
# KubeRay Initialization Script
# This script sets up a KubeRay cluster locally using Kind and Helm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

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

# Check uv first — it's the foundation
ensure_uv
echo -e "✅ uv: $(uv --version)"

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

if ! command_exists docker && ! command_exists podman; then
  echo -e "❌ No container runtime found. Please install Docker or Podman."
  missing_deps=1
fi

# Ensure Python dependencies are synced (uv handles venv automatically)
echo -e "Syncing Python dependencies with uv..."
(cd "$PROJECT_ROOT" && uv sync --quiet)
if [ $? -eq 0 ]; then
  echo -e "✅ Python dependencies synced"
else
  echo -e "❌ Failed to sync Python dependencies. Run 'uv sync' manually."
  missing_deps=1
fi

if [ $missing_deps -eq 1 ]; then
  echo -e "\n${RED}Please install the missing dependencies and try again.${NC}"
  exit 1
fi

echo -e "✅ All required prerequisites are installed\n"

# Detect container runtime
detect_container_runtime

# Step 2: Create Kind cluster
echo -e "${YELLOW}Step 2/7: Creating Kind cluster...${NC}"
if kind_cmd get clusters | grep -q "kind"; then
  echo -e "Kind cluster already exists, refreshing kubeconfig..."
  kind_cmd export kubeconfig --name kind
else
  echo -e "Creating Kind cluster with Kubernetes v1.35.0..."
  kind_cmd create cluster --image=kindest/node:v1.35.0 --config="$PROJECT_ROOT/kind-config.yaml"
  if [ $? -eq 0 ]; then
    echo -e "✅ Kind cluster created successfully"
  else
    echo -e "❌ Failed to create Kind cluster"
    exit 1
  fi
fi

# Podman defaults to 2048 PIDs per container. Kind runs all of k8s inside a
# single container, so this limit is shared across ALL system pods + user pods.
# Ray head alone needs ~450 PIDs/threads. Bump the limit to avoid freezes.
if [ "$CONTAINER_RT" = "podman" ]; then
  echo -e "Increasing PID limit on Kind node container..."
  $CONTAINER_RT update --pids-limit 8192 kind-control-plane >/dev/null 2>&1 \
    && echo -e "✅ PID limit set to 8192" \
    || echo -e "⚠️  Could not update PID limit (may need manual fix)"
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

# Step 4: Start Prometheus, Grafana, and MinIO via Compose
echo -e "${YELLOW}Step 4/8: Starting services with ${COMPOSE_CMD}...${NC}"
mkdir -p "$PROJECT_ROOT/data/prometheus" "$PROJECT_ROOT/data/grafana" "$PROJECT_ROOT/data/minio"
chmod 777 "$PROJECT_ROOT/data/prometheus" "$PROJECT_ROOT/data/grafana"

# Stop any brew-managed Grafana/Prometheus that would conflict
if command_exists brew; then
  brew services stop grafana 2>/dev/null || true
  brew services stop prometheus 2>/dev/null || true
fi

(cd "$PROJECT_ROOT" && $COMPOSE_CMD up -d)
echo -e "⏳ Waiting for services to initialize..."
sleep 10

# Connect Prometheus and Grafana to the Kind network so Ray pods can reach them
echo -e "Connecting Prometheus and Grafana to Kind network..."
$CONTAINER_RT network connect kind prometheus_server 2>/dev/null || true
$CONTAINER_RT network connect kind grafana_server 2>/dev/null || true

# Get their IPs on the Kind network
PROMETHEUS_K8S_IP=$($CONTAINER_RT inspect prometheus_server --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')
GRAFANA_K8S_IP=$($CONTAINER_RT inspect grafana_server --format '{{range $k,$v := .NetworkSettings.Networks}}{{if eq $k "kind"}}{{$v.IPAddress}}{{end}}{{end}}')

if [ -z "$PROMETHEUS_K8S_IP" ] || [ -z "$GRAFANA_K8S_IP" ]; then
  echo -e "⚠️  Could not get container IPs on Kind network. Ray Dashboard metrics may not work."
  PROMETHEUS_K8S_IP="localhost"
  GRAFANA_K8S_IP="localhost"
else
  echo -e "✅ Prometheus reachable from pods at ${PROMETHEUS_K8S_IP}:9090"
  echo -e "✅ Grafana reachable from pods at ${GRAFANA_K8S_IP}:3000"
fi
echo

# Step 5: Install Ray cluster
echo -e "${YELLOW}Step 5/8: Installing Ray cluster...${NC}"
ARCH=$(detect_architecture)
echo -e "Detected architecture: ${ARCH}"

HELM_VALUES="$PROJECT_ROOT/helm/ray-cluster-values.yaml"

echo -e "Checking if Ray cluster is already installed..."
if helm list | grep -q "raycluster"; then
  echo -e "Ray cluster already exists, upgrading with current settings..."
  HELM_CMD="upgrade"
else
  HELM_CMD="install"
fi

if [ "$ARCH" = "aarch64" ]; then
  RAY_IMAGE_TAG="2.44.1-py311-aarch64"
else
  RAY_IMAGE_TAG="2.44.1-py311"
fi
echo -e "Deploying Ray cluster for ${ARCH} architecture (image: ${RAY_IMAGE_TAG})..."

# Generate overrides file for arch + metrics service IPs
OVERRIDE_VALUES=$(mktemp)
cat > "$OVERRIDE_VALUES" <<EOF
image:
  tag: ${RAY_IMAGE_TAG}
head:
  containerEnv:
    - name: RAY_GRAFANA_HOST
      value: "http://${GRAFANA_K8S_IP}:3000"
    - name: RAY_PROMETHEUS_HOST
      value: "http://${PROMETHEUS_K8S_IP}:9090"
    - name: RAY_GRAFANA_IFRAME_HOST
      value: "http://localhost:3000"
    - name: RAY_PROMETHEUS_NAME
      value: "Prometheus"
EOF

helm $HELM_CMD raycluster kuberay/ray-cluster --version 1.5.1 \
  -f "$HELM_VALUES" \
  -f "$OVERRIDE_VALUES"
HELM_EXIT=$?
rm -f "$OVERRIDE_VALUES"

if [ $HELM_EXIT -eq 0 ]; then
  echo -e "✅ Ray cluster deployed successfully"
else
  echo -e "❌ Failed to deploy Ray cluster"
  exit 1
fi

# Relax health probes — KubeRay's defaults are too aggressive for a homelab.
# Ray startup takes ~60s; the default readiness probe (delay=10s, failure=10,
# period=5s) fails before Ray is ready, causing unnecessary restarts.
echo -e "Patching health probe timeouts..."
kubectl patch raycluster raycluster-kuberay --type='json' -p="$(cat <<'PROBEPATCH'
[
  {"op": "add", "path": "/spec/headGroupSpec/template/spec/containers/0/livenessProbe", "value": {
    "exec": {"command": ["bash", "-c", "wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success && wget --tries 1 -T 10 -q -O- http://localhost:8265/api/gcs_healthz | grep success"]},
    "initialDelaySeconds": 90, "periodSeconds": 10, "timeoutSeconds": 15, "failureThreshold": 20
  }},
  {"op": "add", "path": "/spec/headGroupSpec/template/spec/containers/0/readinessProbe", "value": {
    "exec": {"command": ["bash", "-c", "wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success && wget --tries 1 -T 10 -q -O- http://localhost:8265/api/gcs_healthz | grep success"]},
    "initialDelaySeconds": 60, "periodSeconds": 10, "timeoutSeconds": 15, "failureThreshold": 20
  }},
  {"op": "add", "path": "/spec/workerGroupSpecs/0/template/spec/containers/0/livenessProbe", "value": {
    "exec": {"command": ["bash", "-c", "wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success"]},
    "initialDelaySeconds": 60, "periodSeconds": 10, "timeoutSeconds": 10, "failureThreshold": 20
  }},
  {"op": "add", "path": "/spec/workerGroupSpecs/0/template/spec/containers/0/readinessProbe", "value": {
    "exec": {"command": ["bash", "-c", "wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success"]},
    "initialDelaySeconds": 30, "periodSeconds": 10, "timeoutSeconds": 10, "failureThreshold": 20
  }}
]
PROBEPATCH
)" >/dev/null 2>&1 && echo -e "✅ Health probes relaxed" || echo -e "⚠️  Could not patch probes (pods may restart once during startup)"
echo

# Step 6: Wait for cluster to be ready
echo -e "${YELLOW}Step 6/8: Waiting for Ray cluster to be ready...${NC}"
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

# Step 7: Verify compose services
echo -e "${YELLOW}Step 7/8: Verifying services...${NC}"
curl -s -o /dev/null http://localhost:9000 && echo -e "✅ MinIO is accessible" || echo -e "⚠️ MinIO may not be accessible"
curl -s -o /dev/null http://localhost:9090/-/healthy && echo -e "✅ Prometheus is healthy" || echo -e "⚠️ Prometheus may not be healthy"
curl -s -o /dev/null http://localhost:3000/api/health && echo -e "✅ Grafana is healthy" || echo -e "⚠️ Grafana may not be healthy"
echo

# Step 8: Setup port forwarding and start Streamlit
echo -e "${YELLOW}Step 8/8: Setting up port forwarding and starting services...${NC}"

# Clean up any existing port forwarding
pkill -f "kubectl port-forward.*raycluster-kuberay-head-svc" 2>/dev/null || true
pkill -f "kuberay-port-forward-loop" 2>/dev/null || true
sleep 2

# Auto-reconnecting port-forward: kubectl port-forward dies when pods restart.
# This loop restarts it automatically so the dashboard stays accessible.
PORT_FORWARD_PORTS="8265:8265 6379:6379 10001:10001 8080:8080"
echo -e "Setting up auto-reconnecting port forwarding (dashboard, GCS, client, metrics)..."

cat > /tmp/kuberay-port-forward-loop.sh <<'PFEOF'
#!/bin/bash
# Auto-reconnecting port-forward loop
while true; do
  kubectl wait --for=condition=ready pod \
    --selector=ray.io/cluster=raycluster-kuberay,ray.io/node-type=head \
    --timeout=120s >/dev/null 2>&1
  kubectl port-forward --address 0.0.0.0 \
    service/raycluster-kuberay-head-svc \
    8265:8265 6379:6379 10001:10001 8080:8080 2>/dev/null
  # port-forward exited (pod restarted?) — wait briefly and retry
  sleep 3
done
PFEOF
chmod +x /tmp/kuberay-port-forward-loop.sh
bash /tmp/kuberay-port-forward-loop.sh &
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
  echo -e "Starting Streamlit app with uv..."
  (cd "$PROJECT_ROOT" && uv run streamlit run streamlit_app/app.py) &
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
echo -e "- ${BLUE}Ray Dashboard:${NC} http://localhost:8265/"
echo -e "- ${BLUE}MinIO Console:${NC} http://localhost:9001/ (credentials from .env)"
echo -e "- ${BLUE}Prometheus:${NC} http://localhost:9090/"
echo -e "- ${BLUE}Grafana:${NC} http://localhost:3000/ (admin/admin)"
echo -e "- ${BLUE}Streamlit Dashboard:${NC} http://localhost:8501/"
echo
echo -e "${YELLOW}KubeRay Cluster Status:${NC}"
kubectl get rayclusters
echo
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay
echo
kubectl get service raycluster-kuberay-head-svc
echo
echo -e "${YELLOW}To submit a job to the Ray cluster:${NC}"
echo -e "uv run ray job submit --address http://localhost:8265 -- python -c \"import pprint; import ray; ray.init(); pprint.pprint(ray.cluster_resources(), sort_dicts=True)\""
echo
echo -e "${YELLOW}Note:${NC} To stop all services, run: make stop"
echo -e "${YELLOW}Note:${NC} To check cluster status, run: make status"
echo -e "${YELLOW}Note:${NC} Port forwarding PID: $RAY_PORT_FORWARD_PID (auto-reconnects on pod restarts)"

# Store PIDs for cleanup
echo $RAY_PORT_FORWARD_PID > /tmp/kuberay-port-forward.pid
if [ ! -z "$STREAMLIT_PID" ]; then
  echo $STREAMLIT_PID > /tmp/kuberay-streamlit.pid
fi

# Wait for Ctrl+C — keep the script alive so port-forward loop keeps running
trap "pkill -f kuberay-port-forward-loop 2>/dev/null; kill $RAY_PORT_FORWARD_PID 2>/dev/null; exit 0" INT TERM
if [ ! -z "$STREAMLIT_PID" ]; then
  wait $STREAMLIT_PID
else
  wait $RAY_PORT_FORWARD_PID
fi
