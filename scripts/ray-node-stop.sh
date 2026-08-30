#!/bin/bash
# Stop the Ray node running on this machine, head or worker.
#
# Only affects this machine: stopping a worker drains it from the cluster,
# stopping the head tears the cluster down.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo -e "${BLUE}${BOLD}=== Stopping the local Ray node ===${NC}"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo -e "No synced environment here, so no Ray node to stop."
  exit 0
fi

"$SCRIPT_DIR/uv-run.sh" ray stop
echo -e "${GREEN}✅ Ray stopped on $(uname -n)${NC}"
