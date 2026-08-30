# Configuration Files

This directory contains configuration files for the metrics and monitoring stack.

## Structure

```
config/
├── prometheus.yml                              # Prometheus scrape configuration
├── floci/
│   └── init/
│       └── ready.d/
│           └── 01-create-buckets.sh            # Creates the S3 buckets on boot
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── prometheus.yml                  # Auto-provision Prometheus datasource
        └── dashboards/
            ├── ray-dashboards.yml              # Dashboard provisioning config
            └── json/
                └── (Ray dashboard JSONs go here)
```

## Prometheus Configuration

**File:** `prometheus.yml`

Configures Prometheus to scrape metrics from Ray:
- **Scrape interval:** 15 seconds
- **Ray target:** `host.docker.internal:8080`
- **Job name:** `ray`

To modify the scrape interval or add more targets, edit this file and restart Prometheus:
```bash
docker compose restart prometheus
```

## Floci Configuration

**File:** `floci/init/ready.d/01-create-buckets.sh`

Floci runs every script under `/etc/floci/init/ready.d` once its S3 API is live;
`docker-compose.yaml` mounts this directory there read-only. The script creates
`app-bucket` and `ray-bucket` and makes `app-bucket` publicly readable — the job
the retired `minio/mc` sidecar used to do.

Two constraints to keep in mind when editing it:

- **It must be idempotent.** Storage is persistent, so the buckets already exist
  on every boot after the first, and a hook that exits non-zero cancels the rest
  of the phase.
- **It relies on the `latest-compat` image**, which is the tag that bundles the
  AWS CLI and boto3. The plain `floci/floci` image has neither.

Credentials and the endpoint are pre-set inside the container, so `aws` commands
in the script need no `--endpoint-url` flag. To re-run it:
```bash
docker compose restart floci
```

## Grafana Configuration

Grafana is configured using provisioning files that automatically set up datasources and dashboards when Grafana starts.

### Datasources

**File:** `grafana/provisioning/datasources/prometheus.yml`

Auto-configures Prometheus as the default datasource. No manual setup needed!

### Dashboards

**File:** `grafana/provisioning/dashboards/ray-dashboards.yml`

Configures Grafana to load dashboard JSONs from the `json/` directory.

To add Ray's default dashboards:

1. Start Ray with metrics enabled:
   ```bash
   ray start --head --metrics-export-port=8080
   ```

2. Copy the dashboard JSONs:
   ```bash
   cp /tmp/ray/session_latest/metrics/grafana/dashboards/*.json \
      config/grafana/provisioning/dashboards/json/
   ```

3. Restart Grafana:
   ```bash
   docker compose restart grafana
   ```

The dashboards will appear in Grafana under the "Ray" folder.

## Customization

### Adding More Scrape Targets

Edit `prometheus.yml` and add more scrape configs:

```yaml
scrape_configs:
  - job_name: 'ray'
    static_configs:
      - targets: ['host.docker.internal:8080']

  - job_name: 'my-service'
    static_configs:
      - targets: ['my-service:9100']
```

### Creating Custom Grafana Dashboards

You can create dashboards in the Grafana UI (http://localhost:3000) and export them as JSON files. Place the exported JSON in `grafana/provisioning/dashboards/json/` for automatic loading on restart.

## Environment Variables

The following environment variables in `.env` affect these configurations:

- `FLOCI_PORT`: Host port for the Floci S3 API (default: 4566)
- `FLOCI_BIND_HOST`: Address the Floci port is published on (default: 127.0.0.1)
- `METRICS_EXPORT_PORT`: Port where Ray exports metrics (default: 8080)
- `PROMETHEUS_PORT`: Port where Prometheus is accessible (default: 9090)
- `GRAFANA_PORT`: Port where Grafana is accessible (default: 3000)

## Troubleshooting

### Prometheus Can't Reach Ray

If you see the "ray" target as "DOWN" in Prometheus (http://localhost:9090/targets):

1. Verify Ray is exporting metrics:
   ```bash
   curl http://localhost:8080/metrics
   ```

2. On Linux, ensure `host.docker.internal` is configured:
   ```yaml
   # In docker-compose.yaml
   prometheus:
     extra_hosts:
       - "host.docker.internal:host-gateway"
   ```

### Grafana Dashboards Not Loading

1. Check the dashboard JSON files exist:
   ```bash
   ls grafana/provisioning/dashboards/json/
   ```

2. Check Grafana logs:
   ```bash
   docker compose logs grafana
   ```

3. Verify provisioning config is mounted correctly:
   ```bash
   docker compose exec grafana ls /etc/grafana/provisioning/dashboards/
   ```

## References

- [Prometheus Configuration Docs](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Grafana Provisioning Docs](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Ray Metrics Docs](https://docs.ray.io/en/latest/cluster/metrics.html)
