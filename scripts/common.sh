#!/bin/bash
# Common utilities shared across scripts

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ensure PATH includes common install locations for podman/docker
# Podman Desktop installs to /opt/podman/bin, Homebrew to /opt/homebrew/bin
for _p in /opt/podman/bin /opt/homebrew/bin /usr/local/bin; do
  if [ -d "$_p" ] && [[ ":$PATH:" != *":$_p:"* ]]; then
    export PATH="$_p:$PATH"
  fi
done
unset _p

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running on a specific port
is_port_in_use() {
  if command_exists lsof; then
    lsof -i :"$1" >/dev/null 2>&1
  elif command_exists ss; then
    ss -tlnp | grep -q ":$1 "
  else
    netstat -an | grep -q "LISTEN.*:$1 "
  fi
  return $?
}

# Ensure uv is available
ensure_uv() {
  if ! command_exists uv; then
    echo -e "${RED}uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    echo -e "${RED}See: https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
  fi
}

# Run a Python command through uv
uv_run() {
  (cd "$PROJECT_ROOT" && uv run "$@")
}

# Detect container runtime (docker or podman)
detect_container_runtime() {
  if command_exists podman; then
    CONTAINER_RT="podman"
    # Check compose options: podman-compose (standalone), podman compose (plugin),
    # or docker-compose (standalone binary — works with podman via DOCKER_HOST)
    if command_exists podman-compose; then
      COMPOSE_CMD="podman-compose"
    elif podman compose version >/dev/null 2>&1; then
      COMPOSE_CMD="podman compose"
    elif command_exists docker-compose; then
      # Standalone docker-compose binary works with podman's socket
      COMPOSE_CMD="docker-compose"
    else
      echo -e "${RED}No compose tool found for Podman.${NC}"
      echo -e "${RED}Install one of:${NC}"
      echo -e "${RED}  - uv tool install podman-compose${NC}"
      echo -e "${RED}  - brew install docker-compose${NC}"
      exit 1
    fi
  elif command_exists docker; then
    CONTAINER_RT="docker"
    if docker compose version >/dev/null 2>&1; then
      COMPOSE_CMD="docker compose"
    elif command_exists docker-compose; then
      COMPOSE_CMD="docker-compose"
    else
      echo -e "${RED}docker compose plugin not found. Install Docker Desktop or the compose plugin.${NC}"
      exit 1
    fi
  else
    echo -e "${RED}No container runtime found. Please install Docker or Podman.${NC}"
    exit 1
  fi
  # Tell Kind to use Podman when Docker isn't available
  if [ "$CONTAINER_RT" = "podman" ]; then
    export KIND_EXPERIMENTAL_PROVIDER=podman
  fi

  echo -e "Using container runtime: ${GREEN}${CONTAINER_RT}${NC} (compose: ${COMPOSE_CMD})"
}

# Wrapper for kind commands — sets provider automatically
kind_cmd() {
  if [ "$CONTAINER_RT" = "podman" ]; then
    KIND_EXPERIMENTAL_PROVIDER=podman kind "$@"
  else
    kind "$@"
  fi
}
