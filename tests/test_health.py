"""Tests for service health checks and disk reporting."""

import socket

import pytest

from streamlit_app import health


@pytest.fixture
def listening_port():
    """Bind a real localhost socket and yield its port."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    yield server.getsockname()[1]
    server.close()


def test_is_port_open_true_for_listening_socket(listening_port):
    assert health.is_port_open("127.0.0.1", listening_port) is True


def test_is_port_open_false_for_closed_port():
    # Port 1 on localhost is reserved and never bound by the test environment.
    assert health.is_port_open("127.0.0.1", 1, timeout=0.25) is False


def test_is_port_open_false_for_unresolvable_host():
    assert health.is_port_open("invalid.host.invalid", 4566, timeout=0.25) is False


def test_port_from_env_reads_the_variable(monkeypatch):
    monkeypatch.setenv("FLOCI_PORT", "5566")
    assert health.port_from_env("FLOCI_PORT", 4566) == 5566


def test_port_from_env_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("FLOCI_PORT", raising=False)
    assert health.port_from_env("FLOCI_PORT", 4566) == 4566


def test_port_from_env_falls_back_on_garbage(monkeypatch):
    """A typo in .env must not crash the dashboard on import."""
    monkeypatch.setenv("FLOCI_PORT", "not-a-number")
    assert health.port_from_env("FLOCI_PORT", 4566) == 4566


@pytest.mark.parametrize(
    ("check", "expected_port"),
    [
        (health.is_ray_running, health.RAY_DASHBOARD_PORT),
        (health.is_floci_running, health.FLOCI_PORT),
    ],
)
def test_service_checks_use_expected_ports(monkeypatch, check, expected_port):
    seen = {}

    def fake_is_port_open(host, port, timeout=health.DEFAULT_TIMEOUT):
        seen["host"], seen["port"] = host, port
        return True

    monkeypatch.setattr(health, "is_port_open", fake_is_port_open)
    assert check() is True
    assert seen == {"host": health.DEFAULT_HOST, "port": expected_port}


@pytest.mark.parametrize(
    ("running", "expected"), [(True, "✅ Online"), (False, "❌ Offline")]
)
def test_status_label(running, expected):
    assert health.status_label(running) == expected


def test_format_disk_usage_reports_whole_gibibytes():
    assert health.format_disk_usage(100 * health.GIB, 25 * health.GIB) == (
        "25 GB free out of 100 GB"
    )


def test_format_disk_usage_truncates_partial_gibibytes():
    assert health.format_disk_usage(health.GIB * 3 // 2, health.GIB // 2) == (
        "0 GB free out of 1 GB"
    )


def test_get_disk_usage_returns_summary(tmp_path):
    assert "GB free out of" in health.get_disk_usage(str(tmp_path))


def test_get_disk_usage_reports_errors(monkeypatch):
    def boom(path):
        raise OSError("no such path")

    monkeypatch.setattr(health.shutil, "disk_usage", boom)
    assert health.get_disk_usage("/nope") == "Error: no such path"
