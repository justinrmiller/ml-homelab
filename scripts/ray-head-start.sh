#!/bin/bash
# Start a native Ray head on this machine for a multi-machine cluster.
#
# This is a different topology from `make start`, which runs Ray inside Kind.
# Kind publishes its ports on loopback and Ray needs a wide, bidirectional port
# range between every node, so a Kind-hosted head cannot accept workers from
# other machines. For a multi-machine cluster the head runs natively, here.
#
# Usage:
#   scripts/ray-head-start.sh
#
# Ports are pinned rather than random so a LAN firewall rule can be written
# once; see docs/multi-machine.md for the list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GCS_PORT="$(env_value RAY_HEAD_PORT 6379)"
DASHBOARD_PORT="$(env_value DASHBOARD_PORT 8265)"
CLIENT_PORT="$(env_value HEAD_NODE_PORT 10001)"
METRICS_PORT="$(env_value METRICS_EXPORT_PORT 8080)"
NODE_MANAGER_PORT="$(env_value RAY_NODE_MANAGER_PORT 6380)"
OBJECT_MANAGER_PORT="$(env_value RAY_OBJECT_MANAGER_PORT 6381)"
# Ray otherwise grabs 10002-19999 for worker ports, which swallows any custom
# Ray Client port and makes "open these ports" impossible to state precisely.
# A narrow pinned range keeps the firewall rule in docs/multi-machine.md honest.
MIN_WORKER_PORT="$(env_value RAY_MIN_WORKER_PORT 11000)"
MAX_WORKER_PORT="$(env_value RAY_MAX_WORKER_PORT 11999)"

echo -e "${BLUE}${BOLD}=== Starting Ray head ===${NC}"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo -e "${RED}❌ No synced environment found${NC}"
  echo -e "   Run ${BLUE}scripts/ray-node-setup.sh${NC} first."
  exit 1
fi

if is_port_in_use "$GCS_PORT"; then
  echo -e "${RED}❌ Port ${GCS_PORT} is already in use${NC}"
  echo -e "   Something else holds the GCS port — often a leftover"
  echo -e "   \`kubectl port-forward\` from the Kind topology, or a previous head."
  echo -e "   Inspect it with: ${BLUE}lsof -nP -iTCP:${GCS_PORT} -sTCP:LISTEN${NC}"
  echo -e "   Stop a previous head with: ${BLUE}make node-stop${NC}"
  exit 1
fi

# Ray refuses to form a multi-node cluster on macOS unless this is set:
#   "Multi-node Ray clusters are not supported on Windows and OSX."
# Ray treats the configuration as unsupported rather than broken, and it does
# work on a LAN, but that is the vendor's own caveat and worth knowing.
if [ "$(uname -s)" = "Darwin" ]; then
  export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
  echo -e "${YELLOW}macOS: enabling multi-node support, which Ray classes as unsupported.${NC}"
  echo -e "${YELLOW}It works on a LAN, but Linux nodes are the better-tested path.${NC}\n"
fi

# 0.0.0.0 on purpose: workers on other machines have to reach the GCS. This is
# the point at which the cluster becomes reachable from the LAN, so run it on a
# network you trust — Ray has no authentication of its own.
echo -e "Binding the GCS on all interfaces so remote workers can join."
echo -e "${YELLOW}Ray has no built-in authentication; use this on a trusted network only.${NC}\n"

"$SCRIPT_DIR/uv-run.sh" ray start --head \
  --port "$GCS_PORT" \
  --node-manager-port "$NODE_MANAGER_PORT" \
  --object-manager-port "$OBJECT_MANAGER_PORT" \
  --min-worker-port "$MIN_WORKER_PORT" \
  --max-worker-port "$MAX_WORKER_PORT" \
  --ray-client-server-port "$CLIENT_PORT" \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "$DASHBOARD_PORT" \
  --metrics-export-port "$METRICS_PORT" \
  --disable-usage-stats || {
    echo -e "${RED}❌ Ray head failed to start${NC}"
    exit 1
  }

# Prefer the LAN address over the loopback one Ray prints, since the whole
# point of this topology is that other machines connect to it.
HEAD_IP="$(ipconfig getifaddr en0 2>/dev/null \
  || hostname -I 2>/dev/null | awk '{print $1}' \
  || echo "<this-machine-ip>")"

echo
echo -e "${GREEN}${BOLD}=== Ray head running ===${NC}"
echo -e "- ${BLUE}Dashboard:${NC} http://localhost:${DASHBOARD_PORT}/"
echo -e "- ${BLUE}GCS address:${NC} ${HEAD_IP}:${GCS_PORT}"
echo
echo -e "On each worker machine, from a checkout at the same commit:"
echo -e "  ${BLUE}scripts/ray-node-setup.sh${NC}"
echo -e "  ${BLUE}make worker-start HEAD=${HEAD_IP}${NC}"
