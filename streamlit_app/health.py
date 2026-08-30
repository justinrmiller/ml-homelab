"""Health and capacity checks for the services backing the homelab."""

import os
import shutil
import socket

DEFAULT_FLOCI_PORT = 4566


def port_from_env(name: str, default: int) -> int:
    """Read a TCP port from the environment.

    Falls back to ``default`` when the variable is unset, empty, or not a
    number, so a typo in ``.env`` degrades to the default instead of crashing
    the dashboard on import.

    Args:
        name: Environment variable to read.
        default: Port to use when the variable is unusable.

    Returns:
        The configured port, or ``default``.
    """
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# docker-compose.yaml publishes ${FLOCI_PORT:-4566}; read the same variable so
# a non-default port does not leave the dashboard reporting Floci as offline.
FLOCI_PORT = port_from_env("FLOCI_PORT", DEFAULT_FLOCI_PORT)
RAY_DASHBOARD_PORT = 8265
DEFAULT_HOST = "localhost"
DEFAULT_TIMEOUT = 2.0

GIB = 1024**3


def is_port_open(
    host: str = DEFAULT_HOST,
    port: int = RAY_DASHBOARD_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Report whether a TCP connection to ``host:port`` can be established.

    Args:
        host: Hostname to connect to.
        port: TCP port to connect to.
        timeout: Connection timeout in seconds.

    Returns:
        True when the port accepts a connection, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_ray_running(host: str = DEFAULT_HOST, port: int = RAY_DASHBOARD_PORT) -> bool:
    """Check whether the Ray dashboard is reachable."""
    return is_port_open(host, port)


def is_floci_running(host: str = DEFAULT_HOST, port: int = FLOCI_PORT) -> bool:
    """Check whether the Floci S3 endpoint is reachable."""
    return is_port_open(host, port)


def status_label(running: bool) -> str:
    """Render a service status as a human readable label."""
    return "✅ Online" if running else "❌ Offline"


def format_disk_usage(total_bytes: int, free_bytes: int) -> str:
    """Format raw byte counts as a ``N GB free out of M GB`` summary."""
    return f"{free_bytes // GIB} GB free out of {total_bytes // GIB} GB"


def get_disk_usage(path: str = "/") -> str:
    """Summarize free and total disk space for ``path``.

    Returns:
        A human readable summary, or an ``Error: ...`` string when the path
        cannot be inspected.
    """
    try:
        total, _used, free = shutil.disk_usage(path)
    except OSError as exc:
        return f"Error: {exc}"
    return format_disk_usage(total, free)
