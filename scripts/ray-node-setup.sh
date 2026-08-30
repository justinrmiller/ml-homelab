#!/bin/bash
# Prepare this machine to run a Ray node (head or worker) for the homelab.
#
# Ray refuses to join a cluster whose Ray version differs from the head's, and
# the Python versions have to line up too. Rather than asking each machine to
# get that right by hand, every node syncs the *same* pinned environment:
# .python-version pins the interpreter and uv.lock pins ray to an exact build.
# Two machines on the same commit of this repo therefore agree by construction.
#
# Usage:
#   scripts/ray-node-setup.sh
#
# Override the PyTorch build with TORCH_EXTRA=cpu|gpu (see scripts/torch-extra.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo -e "${BLUE}${BOLD}=== Ray node setup ===${NC}"
echo -e "Host: $(uname -n)  OS: $(uname -s)  Arch: $(uname -m)\n"

# Step 1: uv
echo -e "${YELLOW}Step 1/3: Checking uv...${NC}"
if ! command_exists uv; then
  echo -e "${RED}❌ uv is not installed${NC}"
  echo -e "   uv manages the pinned Python and Ray versions this node needs."
  echo -e "   Install it, then re-run this script:"
  echo -e "     ${BLUE}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
  echo -e "   (or: brew install uv)"
  exit 1
fi
echo -e "✅ uv: $(uv --version)"
echo

# Step 2: sync the pinned environment
echo -e "${YELLOW}Step 2/3: Syncing the pinned environment...${NC}"
TORCH_EXTRA_RESOLVED="$("$SCRIPT_DIR/torch-extra.sh")"
echo -e "PyTorch build for this machine: ${BOLD}${TORCH_EXTRA_RESOLVED}${NC}"

# --frozen: install exactly what uv.lock pins. Without it uv may re-resolve and
# quietly land a different Ray build here than the head is running.
if (cd "$PROJECT_ROOT" && uv sync --frozen --extra "$TORCH_EXTRA_RESOLVED"); then
  echo -e "✅ Environment synced"
else
  echo -e "${RED}❌ uv sync failed${NC}"
  exit 1
fi
echo

# Step 3: report the identity every other node has to match
echo -e "${YELLOW}Step 3/3: Version report...${NC}"
RAY_LOCAL="$(ray_local_version)"
if [ -z "$RAY_LOCAL" ]; then
  echo -e "${RED}❌ Could not import ray from the synced environment${NC}"
  exit 1
fi
echo -e "✅ Python:     $("$SCRIPT_DIR/uv-run.sh" python -c 'import platform; print(platform.python_version())')"
echo -e "✅ Ray:        ${RAY_LOCAL% *} (commit ${RAY_LOCAL#* })"
echo -e "✅ Repo commit: $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo
echo -e "${GREEN}${BOLD}Ready.${NC} Every node in the cluster must report the same Ray"
echo -e "version and commit above — check out the same commit of this repo on each."
echo
echo -e "Next:"
echo -e "  On the head machine:   ${BLUE}make head-start${NC}"
echo -e "  On each worker:        ${BLUE}make worker-start HEAD=<head-ip>${NC}"
