#!/bin/bash
# Start or stop the Streamlit dashboard.
#
# Independent of the Ray topology: the dashboard's S3 tab talks to Floci, and
# its Ray panel just probes a port, so it is useful on its own.
#
# Usage:
#   scripts/streamlit.sh start
#   scripts/streamlit.sh stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PID_FILE=/tmp/kuberay-streamlit.pid
STREAMLIT_PORT="$(env_value STREAMLIT_PORT 8501)"

case "${1:-start}" in
  start)
    if is_port_in_use "$STREAMLIT_PORT"; then
      echo -e "Streamlit is already running on port ${STREAMLIT_PORT}."
      exit 0
    fi
    echo -e "${BLUE}Starting Streamlit on port ${STREAMLIT_PORT}...${NC}"
    # Via uv-run.sh, not a bare `uv run`: the torch extra has to match this
    # machine or the environment resolves without torch.
    # Redirect the whole subshell, not just the command inside it. Otherwise
    # the backgrounded subshell keeps this script's stdout open and anything
    # capturing our output (make in a pipeline, CI) waits on a process that
    # never exits.
    (cd "$PROJECT_ROOT" && exec "$SCRIPT_DIR/uv-run.sh" streamlit run streamlit_app/app.py \
      --server.port "$STREAMLIT_PORT") </dev/null >/tmp/streamlit.log 2>&1 &
    echo $! > "$PID_FILE"
    echo -e "${GREEN}✅ Streamlit starting${NC} — http://localhost:${STREAMLIT_PORT}/"
    echo -e "   Logs: /tmp/streamlit.log"
    ;;
  stop)
    if ! is_port_in_use "$STREAMLIT_PORT"; then
      echo -e "Streamlit was not running."
      rm -f "$PID_FILE"
      exit 0
    fi

    # `uv run streamlit ...` leaves two processes: the uv wrapper and the
    # server it execs. Signalling one does not necessarily take the other
    # down, so terminate both and then confirm against the port rather than
    # trusting kill's exit status — a stop that reports success while the app
    # is still serving is worse than one that admits it failed.
    [ -f "$PID_FILE" ] && kill "$(cat "$PID_FILE")" 2>/dev/null
    pkill -f "streamlit run streamlit_app/app.py" 2>/dev/null

    for _ in 1 2 3 4 5 6 7 8 9 10; do
      is_port_in_use "$STREAMLIT_PORT" || break
      sleep 0.5
    done

    if is_port_in_use "$STREAMLIT_PORT"; then
      echo -e "${YELLOW}Streamlit ignored SIGTERM; sending SIGKILL...${NC}"
      pkill -9 -f "streamlit run streamlit_app/app.py" 2>/dev/null
      for _ in 1 2 3 4 5 6; do
        is_port_in_use "$STREAMLIT_PORT" || break
        sleep 0.5
      done
    fi

    rm -f "$PID_FILE"
    if is_port_in_use "$STREAMLIT_PORT"; then
      echo -e "${RED}❌ Port ${STREAMLIT_PORT} is still in use${NC}"
      echo -e "   Inspect it with: ${BLUE}lsof -nP -iTCP:${STREAMLIT_PORT} -sTCP:LISTEN${NC}"
      exit 1
    fi
    echo -e "${GREEN}✅ Streamlit stopped${NC}"
    ;;
  *)
    echo -e "${RED}Usage: scripts/streamlit.sh {start|stop}${NC}"
    exit 1
    ;;
esac
