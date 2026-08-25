"""Tests for the standalone Ray examples."""

import pytest
from ray.job_submission import JobStatus

from examples import hello_ray_job, ray_job_example
from streamlit_app import job_runner
from tests.conftest import FakeJobClient


def test_hello_world_returns_greeting():
    assert hello_ray_job.hello_world() == "hello world"


def test_hello_main_initializes_ray_and_prints(monkeypatch, capsys):
    monkeypatch.setattr(hello_ray_job.ray, "init", lambda: None)
    monkeypatch.setattr(hello_ray_job.ray, "get", lambda ref: "hello world")
    monkeypatch.setattr(
        hello_ray_job.hello_world_remote, "remote", lambda: object(), raising=False
    )

    hello_ray_job.main()

    assert "hello world" in capsys.readouterr().out


@pytest.fixture
def fake_client(monkeypatch):
    """Route ``create_client`` to a scripted Jobs API double."""
    client = FakeJobClient([JobStatus.RUNNING, JobStatus.SUCCEEDED])
    monkeypatch.setattr(ray_job_example, "create_client", lambda address: client)
    monkeypatch.setattr(job_runner.time, "sleep", lambda seconds: None)
    return client


def test_example_submits_from_the_repository_root(fake_client, capsys):
    ray_job_example.main()

    submitted = fake_client.submitted[0]
    assert submitted["entrypoint"] == "python examples/hello_ray_job.py"
    assert submitted["runtime_env"] == {"working_dir": "./"}

    out = capsys.readouterr().out
    assert "Job submitted: job_abc123" in out
    assert "Job Logs:" in out


def test_example_polls_until_the_job_finishes(fake_client, capsys):
    ray_job_example.main()

    assert fake_client.status_calls == 2
    assert capsys.readouterr().out.count("Status:") == 2
