# Docker Setup Guide for ML Homelab

## Overview

This ML Homelab uses Docker Compose to manage containerized services, specifically MinIO for object storage. Docker Compose simplifies the management of multi-container applications through a single YAML configuration file.

## Current Docker Setup

### Services

#### MinIO Object Storage
- **Container**: `minio_server`
- **Image**: `minio/minio:latest`
- **Ports**: 
  - `9000`: MinIO API
  - `9001`: MinIO Console
- **Volumes**: `./data/minio:/data`
- **Environment**: Configurable via `.env` file

#### MinIO Initialization
- **Container**: `minio_init`
- **Image**: `minio/mc:latest`
- **Purpose**: Creates default buckets (`app-bucket`, `ray-bucket`)
- **Dependencies**: Waits for MinIO service health check

## Docker Compose Commands

### Basic Operations

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View running services
docker compose ps

# View service logs
docker compose logs -f

# View logs for specific service
docker compose logs -f minio

# Restart a service
docker compose restart minio
```

### Advanced Operations

```bash
# Rebuild and start services
docker compose up --build -d

# Scale services (if supported)
docker compose up --scale minio=2

# Run one-off commands
docker compose run minio-init mc --help

# Execute commands in running container
docker compose exec minio sh
```

## Configuration Management

### Environment Variables

Configure services through `.env` file:

```env
# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_DEFAULT_BUCKETS=app-bucket,ray-bucket
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect ml-homelab_minio-data

# Remove unused volumes
docker volume prune
```

## Best Practices for ML Projects

### 1. Service Organization
- **Separate services** for different components (storage, compute, web interface)
- **Use dependencies** with `depends_on` to ensure proper startup order
- **Implement health checks** for service reliability

### 2. Data Management
- **Persistent volumes** for model weights and datasets
- **Named volumes** for better organization
- **Backup strategies** for critical data

### 3. Security
- **Non-root users** in containers where possible
- **Environment variables** for sensitive configuration
- **Network isolation** between services
- **Regular image updates** for security patches

### 4. Development Workflow
- **Override files** for development-specific configuration
- **Live reloading** with bind mounts during development
- **Multi-stage builds** for optimized production images

## Current Implementation Analysis

### Strengths
✅ **Health checks** implemented for MinIO
✅ **Environment configuration** via `.env` file
✅ **Service dependencies** properly configured
✅ **Persistent storage** for data
✅ **Automatic bucket creation** on startup

### Potential Improvements
🔄 **Non-root user** implementation
🔄 **Resource limits** for containers
🔄 **Backup volumes** for critical data
🔄 **Development overrides** for local development

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status
docker compose ps

# View detailed logs
docker compose logs [service-name]

# Check container resources
docker stats
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep [port]

# Modify ports in docker-compose.yaml or .env
```

#### Storage Issues
```bash
# Check disk space
df -h

# Clean up unused resources
docker system prune -a
```

### Service Health
```bash
# Check MinIO health
curl -f http://localhost:9000/minio/health/live

# Check container health status
docker compose ps --format "table {{.Service}}\t{{.Status}}"
```

## Integration with Ray

The Docker setup complements the Ray cluster by:
- **Providing object storage** for Ray jobs
- **Persisting data** across Ray cluster restarts
- **Enabling distributed storage** for ML workloads

## Next Steps

1. **Review logs** regularly for service health
2. **Monitor resource usage** with `docker stats`
3. **Implement backup strategy** for MinIO data
4. **Consider additional services** (monitoring, logging)
5. **Optimize container images** for production use

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MinIO Docker Setup](https://min.io/docs/minio/container/index.html)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)