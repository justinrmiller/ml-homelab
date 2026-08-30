#!/bin/bash
# Join this machine to a Ray head as a worker node.
#
# Usage:
#   scripts/ray-worker-start.sh <head-ip-or-host>[:port]
#   RAY_HEAD_ADDRESS=192.168.1.10 scripts/ray-worker-start.sh
#
# The head address may also live in .env as RAY_HEAD_ADDRESS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

GCS_PORT="$(env_value RAY_HEAD_PORT 6379)"
DASHBOARD_PORT="$(env_value DASHBOARD_PORT 8265)"
NODE_MANAGER_PORT="$(env_value RAY_NODE_MANAGER_PORT 6380)"
OBJECT_MANAGER_PORT="$(env_value RAY_OBJECT_MANAGER_PORT 6381)"
# Ray otherwise grabs 10002-19999 for worker ports, which swallows any custom
# Ray Client port and makes "open these ports" impossible to state precisely.
# A narrow pinned range keeps the firewall rule in docs/multi-machine.md honest.
MIN_WORKER_PORT="$(env_value RAY_MIN_WORKER_PORT 11000)"
MAX_WORKER_PORT="$(env_value RAY_MAX_WORKER_PORT 11999)"

HEAD_ARG="${1:-$(env_value RAY_HEAD_ADDRESS "")}"
if [ -z "$HEAD_ARG" ]; then
  echo -e "${RED}❌ No head address given${NC}"
  echo -e "   Usage: ${BLUE}scripts/ray-worker-start.sh <head-ip>${NC}"
  echo -e "   or set RAY_HEAD_ADDRESS in .env"
  exit 1
fi

# Accept a bare host or host:port.
case "$HEAD_ARG" in
  *:*) HEAD_ADDRESS="$HEAD_ARG"; HEAD_HOST="${HEAD_ARG%:*}" ;;
  *)   HEAD_ADDRESS="${HEAD_ARG}:${GCS_PORT}"; HEAD_HOST="$HEAD_ARG" ;;
esac

echo -e "${BLUE}${BOLD}=== Joining Ray cluster at ${HEAD_ADDRESS} ===${NC}"
echo -e "Host: $(uname -n)  OS: $(uname -s)  Arch: $(uname -m)\n"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo -e "${RED}❌ No synced environment found${NC}"
  echo -e "   Run ${BLUE}scripts/ray-node-setup.sh${NC} first."
  exit 1
fi

# One Ray node per machine. Head and worker pin the same node-manager and
# object-manager ports — correct across machines, but on a single host the
# second raylet blocks forever trying to bind ports the first one holds, with
# no error. Fail fast instead of hanging.
if is_port_in_use "$NODE_MANAGER_PORT" || is_port_in_use "$OBJECT_MANAGER_PORT"; then
  echo -e "${RED}❌ A Ray node is already running on this machine${NC}"
  echo -e "   Ports ${NODE_MANAGER_PORT}/${OBJECT_MANAGER_PORT} are in use, so a second raylet here"
  echo -e "   would hang rather than fail. Run one node per machine."
  echo -e "   Stop the existing node with ${BLUE}make node-stop${NC}, or run this"
  echo -e "   worker on a different machine."
  exit 1
fi

# Pre-flight the version match. Ray would reject a mismatched worker anyway,
# but it does so from deep inside the raylet with a message that buries the
# cause; catching it here names the fix instead.
echo -e "${YELLOW}Checking version parity with the head...${NC}"
LOCAL_VERSION="$(ray_local_version)"
if [ -z "$LOCAL_VERSION" ]; then
  echo -e "${RED}❌ Could not import ray locally — re-run scripts/ray-node-setup.sh${NC}"
  exit 1
fi

if HEAD_VERSION="$(ray_head_version "http://${HEAD_HOST}:${DASHBOARD_PORT}")" \
   && [ -n "${HEAD_VERSION// /}" ]; then
  if [ "$LOCAL_VERSION" = "$HEAD_VERSION" ]; then
    echo -e "✅ Ray ${LOCAL_VERSION% *} matches the head"
  else
    echo -e "${RED}❌ Version mismatch — this worker cannot join${NC}"
    echo -e "   head:  ${HEAD_VERSION}"
    echo -e "   local: ${LOCAL_VERSION}"
    echo -e "   Check out the same commit of ml-homelab on both machines and"
    echo -e "   re-run ${BLUE}scripts/ray-node-setup.sh${NC} here."
    exit 1
  fi
else
  # Not fatal: the dashboard may simply not be exposed. Ray itself still
  # enforces the version match when the raylet connects.
  echo -e "⚠️  Could not reach the head dashboard on port ${DASHBOARD_PORT}"
  echo -e "   Skipping the pre-flight check; Ray will still reject a mismatch."
fi
echo

# Ray refuses to form a multi-node cluster on macOS unless this is set:
#   "Multi-node Ray clusters are not supported on Windows and OSX."
# Ray treats the configuration as unsupported rather than broken, and it does
# work on a LAN, but that is the vendor's own caveat and worth knowing.
if [ "$(uname -s)" = "Darwin" ]; then
  export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
  echo -e "${YELLOW}macOS: enabling multi-node support, which Ray classes as unsupported.${NC}"
  echo -e "${YELLOW}It works on a LAN, but Linux nodes are the better-tested path.${NC}\n"
fi

"$SCRIPT_DIR/uv-run.sh" ray start \
  --address "$HEAD_ADDRESS" \
  --node-manager-port "$NODE_MANAGER_PORT" \
  --object-manager-port "$OBJECT_MANAGER_PORT" \
  --min-worker-port "$MIN_WORKER_PORT" \
  --max-worker-port "$MAX_WORKER_PORT" \
  --disable-usage-stats || {
    echo -e "${RED}❌ Failed to join the cluster${NC}"
    echo -e "   Check that ${HEAD_ADDRESS} is reachable from here:"
    echo -e "     ${BLUE}nc -vz ${HEAD_HOST} ${GCS_PORT}${NC}"
    echo -e "   and that the ports in docs/multi-machine.md are open both ways."
    exit 1
  }

echo
echo -e "${GREEN}${BOLD}=== Joined ===${NC}"
echo -e "Confirm from the head with ${BLUE}make node-status${NC},"
echo -e "or watch the node appear at http://${HEAD_HOST}:${DASHBOARD_PORT}/#/cluster"
