"""Health and capacity checks for the services backing the homelab."""

import shutil
import socket

MINIO_PORT = 9000
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


def is_minio_running(host: str = DEFAULT_HOST, port: int = MINIO_PORT) -> bool:
    """Check whether the MinIO S3 endpoint is reachable."""
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
