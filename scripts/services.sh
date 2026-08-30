#!/bin/bash
# Manage the Compose services: Floci (S3), Prometheus, and Grafana.
#
# These are independent of whichever Ray topology you run, which is why they
# live behind their own targets rather than only inside kuberay-init.sh.
# kuberay-init.sh calls this script too, so there is one implementation.
#
# Usage:
#   scripts/services.sh up     [service...]
#   scripts/services.sh down
#   scripts/services.sh status
#   scripts/services.sh logs   [service...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-up}"
shift 2>/dev/null || true

detect_container_runtime
resolve_floci_port

# The bind mounts have to exist and be writable before the containers claim
# them, or the runtime creates them root-owned.
prepare_data_dirs() {
  mkdir -p "$PROJECT_ROOT/data/prometheus" "$PROJECT_ROOT/data/grafana" "$PROJECT_ROOT/data/floci"
  chmod 777 "$PROJECT_ROOT/data/prometheus" "$PROJECT_ROOT/data/grafana" "$PROJECT_ROOT/data/floci"
}

case "$ACTION" in
  up)
    echo -e "${BLUE}${BOLD}=== Starting services ===${NC}"
    prepare_data_dirs
    if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
      echo -e "Creating .env from .env.example..."
      cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    fi
    (cd "$PROJECT_ROOT" && $COMPOSE_CMD up -d "$@") || exit 1
    echo -e "${GREEN}✅ Services started${NC}"
    ;;
  down)
    echo -e "${BLUE}${BOLD}=== Stopping services ===${NC}"
    if $CONTAINER_RT ps --format "{{.Names}}" | grep -qE "floci|prometheus|grafana"; then
      (cd "$PROJECT_ROOT" && $COMPOSE_CMD down) || exit 1
      echo -e "${GREEN}✅ Services stopped${NC}"
    else
      echo -e "No service containers running."
    fi
    ;;
  status)
    echo -e "${BLUE}${BOLD}=== Service status ===${NC}"
    $CONTAINER_RT ps --format "table {{.Names}}\t{{.Status}}" \
      | grep -E "NAMES|floci|prometheus|grafana" || echo "No service containers running."
    echo
    curl -sf -o /dev/null "${FLOCI_URL}/_floci/health" \
      && echo -e "✅ Floci S3 (${FLOCI_PORT}) is ${GREEN}healthy${NC}" \
      || echo -e "❌ Floci S3 (${FLOCI_PORT}) is ${RED}not responding${NC}"
    curl -sf -o /dev/null http://localhost:9090/-/healthy \
      && echo -e "✅ Prometheus (9090) is ${GREEN}healthy${NC}" \
      || echo -e "❌ Prometheus (9090) is ${RED}not responding${NC}"
    curl -sf -o /dev/null http://localhost:3000/api/health \
      && echo -e "✅ Grafana (3000) is ${GREEN}healthy${NC}" \
      || echo -e "❌ Grafana (3000) is ${RED}not responding${NC}"
    ;;
  logs)
    (cd "$PROJECT_ROOT" && $COMPOSE_CMD logs -f "$@")
    ;;
  *)
    echo -e "${RED}Unknown action: ${ACTION}${NC}"
    echo -e "Usage: scripts/services.sh {up|down|status|logs} [service...]"
    exit 1
    ;;
esac
