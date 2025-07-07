# KubeRay Setup Guide for ML Homelab

## Overview

KubeRay is a Ray operator for Kubernetes that simplifies the deployment and management of Ray clusters on Kubernetes. This guide covers how to set up and use KubeRay with your ML Homelab.

## Architecture

The KubeRay setup consists of:
- **Kind cluster**: Local Kubernetes cluster for development
- **KubeRay operator**: Manages Ray cluster lifecycle
- **Ray cluster**: Distributed computing cluster for ML workloads
- **MinIO**: Object storage for data persistence
- **Streamlit**: Web interface for job management

## Prerequisites

### Required Tools
- **Docker**: Container runtime
- **Kind**: Local Kubernetes cluster
- **Helm**: Kubernetes package manager
- **kubectl**: Kubernetes command-line tool
- **Ray**: Python library for distributed computing
- **Streamlit**: Web framework for data applications

### Installation
The setup script will automatically install missing tools using Homebrew:
```bash
brew install kind helm kubectl
```

## Quick Start

### Using Makefile Commands

```bash
# Start KubeRay cluster
make kuberay-start

# Check cluster status
make kuberay-status

# Stop KubeRay cluster
make kuberay-stop
```

### Using Direct Scripts

```bash
# Start KubeRay cluster
scripts/kuberay-init.sh

# Check cluster status
scripts/kuberay-status.sh

# Stop KubeRay cluster
scripts/kuberay-stop.sh
```

### Using Unified Interface

```bash
# Start with KubeRay
scripts/init.sh --kuberay

# Stop KubeRay cluster
scripts/stop.sh --kuberay
```

## Detailed Setup Process

### 1. Cluster Creation
```bash
# Create Kind cluster with specific Kubernetes version
kind create cluster --image=kindest/node:v1.33.1
```

### 2. KubeRay Operator Installation
```bash
# Add KubeRay Helm repository
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install KubeRay operator
helm install kuberay-operator kuberay/kuberay-operator --create-namespace --namespace kuberay-system
```

### 3. Ray Cluster Deployment
```bash
# For ARM architecture (Apple Silicon)
helm install raycluster kuberay/ray-cluster --version 1.3.2 --set 'image.tag=2.47.0-aarch64'

# For x86_64 architecture
helm install raycluster kuberay/ray-cluster --version 1.3.2
```

### 4. Service Access
```bash
# Port forward Ray dashboard
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265

# Access services
# Ray Dashboard: http://localhost:8265/
# MinIO Console: http://localhost:9001/
# Streamlit App: http://localhost:8501/
```

## Configuration

### Environment Variables
Configure KubeRay through `.env` file:

```env
# KubeRay Configuration
KUBERAY_NAMESPACE=default
KUBERAY_CLUSTER_NAME=raycluster-kuberay
KUBERAY_RAY_VERSION=2.47.0
KUBERAY_OPERATOR_VERSION=1.3.2
KIND_CLUSTER_NAME=kind
KIND_NODE_IMAGE=kindest/node:v1.33.1
```

### Cluster Specifications
- **Ray Version**: 2.47.0
- **Kubernetes Version**: 1.33.1
- **Operator Version**: 1.3.2
- **Default Namespace**: default

## Usage

### Submitting Ray Jobs

```bash
# Submit a simple job
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"

# Submit a file-based job
ray job submit --address http://localhost:8265 -- python your_script.py

# Submit with working directory
ray job submit --address http://localhost:8265 --working-dir . -- python your_script.py
```

### Monitoring and Management

```bash
# Check cluster status
kubectl get rayclusters

# Check Ray pods
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay

# Check Ray services
kubectl get service raycluster-kuberay-head-svc

# View Ray logs
kubectl logs -l ray.io/cluster=raycluster-kuberay

# Access Ray dashboard
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

## Status Monitoring

The `scripts/kuberay-status.sh` script provides comprehensive cluster health checking:

### Service Health Checks
- ✅ **Prerequisites**: kubectl, helm, kind, docker availability
- ✅ **Kind Cluster**: Cluster existence and connectivity
- ✅ **KubeRay Operator**: Installation and pod status
- ✅ **Ray Cluster**: Deployment and pod readiness
- ✅ **Port Forwarding**: Active port forwarding processes
- ✅ **MinIO**: Container status and health
- ✅ **Service Accessibility**: HTTP endpoint availability

### Example Status Output
```
=== KubeRay Cluster Status ===
✅ kubectl: Available
✅ Kind cluster: Running
✅ KubeRay operator: Installed
✅ Ray cluster: Installed
✅ Ray dashboard port forward: Active
✅ MinIO containers: Running
✅ Ray dashboard (8265): Port available
✅ Streamlit (8501): Port available
```

## Troubleshooting

### Common Issues

#### 1. Kind Cluster Won't Start
```bash
# Check Docker is running
docker ps

# Delete and recreate cluster
kind delete cluster
kind create cluster --image=kindest/node:v1.33.1
```

#### 2. Ray Pods Not Ready
```bash
# Check pod status
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay

# View pod logs
kubectl logs -l ray.io/cluster=raycluster-kuberay

# Check resource constraints
kubectl describe nodes
```

#### 3. Port Forwarding Issues
```bash
# Kill existing port forwards
killall kubectl

# Restart port forwarding
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

#### 4. Helm Installation Failures
```bash
# Update Helm repositories
helm repo update

# List installed releases
helm list --all-namespaces

# Uninstall and reinstall
helm uninstall raycluster
helm install raycluster kuberay/ray-cluster --version 1.3.2
```

### Resource Management

```bash
# Check cluster resources
kubectl top nodes
kubectl top pods

# Scale Ray cluster (if supported)
kubectl scale raycluster raycluster-kuberay --replicas=3

# View resource quotas
kubectl describe resourcequotas
```

### Cleanup and Reset

```bash
# Complete cleanup
scripts/kuberay-stop.sh

# Manual cleanup if needed
helm uninstall raycluster
helm uninstall kuberay-operator -n kuberay-system
kind delete cluster
```

## Comparison: KubeRay vs Local Ray

| Feature | Local Ray | KubeRay |
|---------|-----------|---------|
| **Setup Complexity** | Simple | Moderate |
| **Resource Management** | Manual | Automatic |
| **Scalability** | Limited | High |
| **Fault Tolerance** | Basic | Advanced |
| **Multi-tenancy** | No | Yes |
| **Monitoring** | Basic | Rich |
| **Storage** | Local | Distributed |

## Best Practices

### 1. Resource Allocation
- Monitor cluster resource usage
- Set appropriate resource limits
- Use node selectors for workload placement

### 2. Job Management
- Use working directories for code dependencies
- Implement proper error handling
- Monitor job execution logs

### 3. Security
- Use namespaces for isolation
- Implement RBAC policies
- Secure service endpoints

### 4. Monitoring
- Regular health checks with `kuberay-status.sh`
- Monitor Ray dashboard metrics
- Set up log aggregation

## Integration with ML Workflows

### 1. Data Pipeline
```python
import ray
from ray import data

# Connect to KubeRay cluster
ray.init(address="ray://localhost:10001")

# Create data pipeline
ds = ray.data.read_parquet("s3://ray-bucket/data/")
processed = ds.map_batches(preprocess_batch)
processed.write_parquet("s3://ray-bucket/processed/")
```

### 2. Model Training
```python
import ray
from ray import train

# Distributed training
trainer = train.Trainer(
    backend="torch",
    num_workers=4,
    resources_per_worker={"CPU": 2, "GPU": 1}
)
trainer.fit()
```

### 3. Hyperparameter Tuning
```python
import ray
from ray import tune

# Distributed hyperparameter tuning
tuner = tune.Tuner(
    trainable=train_model,
    param_space={"lr": tune.loguniform(1e-4, 1e-1)},
    run_config=train.RunConfig(
        resources=train.ScalingConfig(num_workers=4)
    )
)
results = tuner.fit()
```

## Next Steps

1. **Experiment with distributed workloads**
2. **Set up persistent storage for models**
3. **Implement monitoring and alerting**
4. **Explore Ray Serve for model serving**
5. **Scale cluster based on workload demands**

## References

- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [Ray on Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)
- [Kind Documentation](https://kind.sigs.k8s.io/)
- [Helm Charts](https://helm.sh/docs/)