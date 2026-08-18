"""Tests for Ray job submission, runtime env assembly, and polling."""

import pytest
from ray.job_submission import JobStatus

from streamlit_app import job_runner
from streamlit_app.job_runner import JobSpec
from tests.conftest import FakeJobClient


def test_load_runtime_env_without_file():
    assert job_runner.load_runtime_env("./jobs") == {"working_dir": "./jobs"}


def test_load_runtime_env_merges_yaml(tmp_path):
    (tmp_path / "env.yaml").write_text("pip:\n  - torch\n  - numpy\n")
    env = job_runner.load_runtime_env(str(tmp_path), "env.yaml")
    assert env == {"working_dir": str(tmp_path), "pip": ["torch", "numpy"]}


def test_load_runtime_env_ignores_missing_file(tmp_path):
    env = job_runner.load_runtime_env(str(tmp_path), "absent.yaml")
    assert env == {"working_dir": str(tmp_path)}


def test_load_runtime_env_tolerates_empty_yaml(tmp_path):
    (tmp_path / "env.yaml").write_text("")
    env = job_runner.load_runtime_env(str(tmp_path), "env.yaml")
    assert env == {"working_dir": str(tmp_path)}


def test_load_runtime_env_yaml_can_override_working_dir(tmp_path):
    (tmp_path / "env.yaml").write_text("working_dir: /elsewhere\n")
    env = job_runner.load_runtime_env(str(tmp_path), "env.yaml")
    assert env["working_dir"] == "/elsewhere"


def test_shipped_runtime_env_files_exist_and_parse():
    for spec in (*job_runner.TRAINING_JOBS, *job_runner.INFERENCE_JOBS):
        env = job_runner.load_runtime_env(
            job_runner.DEFAULT_WORKING_DIR, spec.runtime_env_file
        )
        assert env["pip"], f"{spec.name} runtime env did not load pip dependencies"


def test_create_client_passes_address(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        job_runner,
        "JobSubmissionClient",
        lambda address: captured.setdefault("address", address),
    )
    job_runner.create_client("http://ray:8265")
    assert captured["address"] == "http://ray:8265"


def test_submit_job_sends_entrypoint_and_runtime_env(tmp_path):
    (tmp_path / "env.yaml").write_text("pip:\n  - torch\n")
    client = FakeJobClient([JobStatus.SUCCEEDED])
    spec = JobSpec("demo", "python demo.py", "env.yaml")

    job_id = job_runner.submit_job(client, spec, working_dir=str(tmp_path))

    assert job_id == "job_abc123"
    assert client.submitted == [
        {
            "entrypoint": "python demo.py",
            "runtime_env": {"working_dir": str(tmp_path), "pip": ["torch"]},
        }
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (JobStatus.SUCCEEDED, True),
        (JobStatus.FAILED, True),
        (JobStatus.STOPPED, True),
        (JobStatus.RUNNING, False),
        (JobStatus.PENDING, False),
    ],
)
def test_is_terminal(status, expected):
    assert job_runner.is_terminal(status) is expected


def test_tail_logs_returns_last_chars():
    assert job_runner.tail_logs("abcdef", limit=3) == "def"


def test_tail_logs_keeps_short_logs_intact():
    assert job_runner.tail_logs("abc", limit=1024) == "abc"


def test_poll_job_returns_immediately_on_terminal_status(sleepless):
    sleeps, sleep = sleepless
    client = FakeJobClient([JobStatus.SUCCEEDED])

    status = job_runner.poll_job(client, "job_abc123", sleep=sleep)

    assert status == JobStatus.SUCCEEDED
    assert sleeps == []
    assert client.status_calls == 1


def test_poll_job_waits_until_terminal(sleepless):
    sleeps, sleep = sleepless
    client = FakeJobClient([JobStatus.PENDING, JobStatus.RUNNING, JobStatus.FAILED])

    status = job_runner.poll_job(client, "job_abc123", poll_interval=2.0, sleep=sleep)

    assert status == JobStatus.FAILED
    assert sleeps == [2.0, 2.0]


def test_poll_job_reports_every_update(sleepless):
    _sleeps, sleep = sleepless
    client = FakeJobClient([JobStatus.RUNNING, JobStatus.SUCCEEDED], logs="x" * 5000)
    updates = []

    job_runner.poll_job(
        client,
        "job_abc123",
        on_update=lambda status, logs: updates.append((status, logs)),
        sleep=sleep,
    )

    assert [status for status, _ in updates] == [
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
    ]
    assert all(len(logs) == job_runner.LOG_TAIL_CHARS for _, logs in updates)


def test_poll_job_skips_log_fetch_without_callback(sleepless):
    _sleeps, sleep = sleepless
    client = FakeJobClient([JobStatus.SUCCEEDED])

    job_runner.poll_job(client, "job_abc123", sleep=sleep)

    assert client.log_calls == 0


@pytest.mark.parametrize(
    ("status", "succeeded", "fragment"),
    [
        (JobStatus.SUCCEEDED, True, "finished successfully"),
        (JobStatus.FAILED, False, "check logs above"),
        (JobStatus.STOPPED, False, "check logs above"),
    ],
)
def test_summarize_result(status, succeeded, fragment):
    ok, message = job_runner.summarize_result("MNIST Tune", status)
    assert ok is succeeded
    assert "MNIST Tune" in message
    assert fragment in message


def test_job_catalog_entries_are_unique_and_named():
    specs = (*job_runner.TRAINING_JOBS, *job_runner.INFERENCE_JOBS)
    names = job_runner.job_names(specs)
    assert names == ["MNIST Tune", "Resnet Inference"]
    assert len(set(names)) == len(names)


def test_job_specs_are_frozen():
    with pytest.raises(AttributeError):
        job_runner.TRAINING_JOBS[0].name = "other"  # ty: ignore[invalid-assignment]
