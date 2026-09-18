"""Tests for the memray Ray profiling job."""

import os
import tempfile

import pytest

from streamlit_app.jobs.memray_profiling import profile_job, workload


class FakeRemote:
    """Stands in for the Ray remote handle, recording the args it is given."""

    def __init__(self):
        self.calls: list[tuple] = []

    def remote(self, *args):
        self.calls.append(args)
        return f"ref-{len(self.calls)}"


def summary(
    pid: int = 101, html: bytes = b"<html></html>"
) -> profile_job.WorkerProfile:
    return profile_job.WorkerProfile(
        pid=pid,
        peak_memory=64 * 1024**2,
        total_allocations=2048,
        retained_bytes=32 * 1024**2,
        flamegraph=html,
    )


@pytest.fixture(autouse=True)
def clean_cache():
    workload.reset_cache()
    yield
    workload.reset_cache()


@pytest.fixture
def stubbed_ray(monkeypatch, tmp_path):
    """Replace every Ray call main() makes, so no cluster is needed."""
    calls: dict[str, object] = {}
    monkeypatch.setattr(profile_job.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(profile_job.ray, "init", lambda: calls.setdefault("init", True))
    monkeypatch.setattr(
        profile_job.ray, "shutdown", lambda: calls.setdefault("shutdown", True)
    )

    remote = FakeRemote()
    monkeypatch.setattr(profile_job, "profile_remote", remote)

    summaries = [summary(101, b"<html>a</html>"), summary(102, b"<html>b</html>")]

    def fake_get(refs):
        refs = list(refs)
        calls["refs"] = refs
        return summaries[: len(refs)]

    monkeypatch.setattr(profile_job.ray, "get", fake_get)
    return calls, remote


def test_capture_path_is_unique_per_call(tmp_path):
    first = profile_job.capture_path(str(tmp_path))
    second = profile_job.capture_path(str(tmp_path))

    assert first != second


def test_capture_path_carries_the_worker_pid(tmp_path):
    """Ray reuses workers, so the pid is what ties a capture to its process."""
    assert f"-{os.getpid()}-" in profile_job.capture_path(str(tmp_path))


def test_capture_path_defaults_to_the_temp_dir():
    assert profile_job.capture_path().startswith(tempfile.gettempdir())


def test_profile_task_reports_real_numbers(tmp_path):
    result = profile_job.profile_task(
        batches=4,
        frames_per_batch=8,
        frame_size=4096,
        index_depth=1,
        leaks=False,
        directory=str(tmp_path),
    )

    assert set(result) == {
        "pid",
        "peak_memory",
        "total_allocations",
        "retained_bytes",
        "flamegraph",
    }
    assert result["peak_memory"] > 0
    assert result["total_allocations"] > 0


def test_profile_task_returns_a_self_contained_flamegraph(tmp_path):
    """The HTML is what makes the profile viewable off the worker."""
    result = profile_job.profile_task(
        batches=2,
        frames_per_batch=4,
        frame_size=1024,
        index_depth=1,
        leaks=False,
        directory=str(tmp_path),
    )

    html = result["flamegraph"]
    assert b"<!DOCTYPE html" in html[:200]
    assert b"ingest_batches" in html


def test_profile_task_writes_a_capture(tmp_path):
    profile_job.profile_task(
        batches=2,
        frames_per_batch=2,
        frame_size=512,
        index_depth=1,
        leaks=False,
        directory=str(tmp_path),
    )

    assert list(tmp_path.glob("memray-*.bin"))


def test_profile_task_resets_the_cache_between_runs(tmp_path):
    """A reused Ray worker would otherwise start from a cache it did not fill."""
    profile_job.profile_task(
        batches=16,
        frames_per_batch=2,
        frame_size=256,
        index_depth=1,
        leaks=False,
        directory=str(tmp_path),
    )

    result = profile_job.profile_task(
        batches=2,
        frames_per_batch=2,
        frame_size=256,
        index_depth=1,
        leaks=False,
        directory=str(tmp_path),
    )

    assert result["retained_bytes"] < 16 * 2 * 256 + 10_000
    assert len([k for k in workload.CACHE if k.startswith("batch-")]) == 2


def test_flamegraph_bytes_renders_both_views(tmp_path):
    capture = profile_job.capture_path(str(tmp_path))
    destination = profile_job.memray.FileDestination(capture, overwrite=True)
    with profile_job.memray.Tracker(destination=destination):
        workload.ingest_batches(batches=2, frames_per_batch=4, frame_size=1024)

    peak = profile_job.flamegraph_bytes(capture)
    leaks = profile_job.flamegraph_bytes(capture, leaks=True)

    assert b"<!DOCTYPE html" in peak[:200]
    assert b"<!DOCTYPE html" in leaks[:200]
    assert b"ingest_batches" in peak
    assert peak != leaks


def test_profile_remote_wraps_the_undecorated_task():
    assert hasattr(profile_job.profile_remote, "remote")


def test_format_report_renders_a_row_per_worker():
    report = profile_job.format_report([summary(7)])

    lines = report.splitlines()
    assert "pid" in lines[0]
    assert len(lines) == 3
    assert "7" in lines[2]
    assert "64.0 MB" in lines[2]
    assert "2,048" in lines[2]


def test_distinct_pids_counts_processes():
    assert profile_job.distinct_pids([summary(1), summary(1), summary(2)]) == 2


def test_write_flamegraphs_writes_one_file_per_task(tmp_path):
    out = tmp_path / "reports"

    paths = profile_job.write_flamegraphs(
        [summary(11, b"<html>a</html>"), summary(12, b"<html>b</html>")], out
    )

    assert len(paths) == 2
    assert paths[0].read_bytes() == b"<html>a</html>"
    assert paths[1].read_bytes() == b"<html>b</html>"


def test_write_flamegraphs_keeps_reused_workers_apart(tmp_path):
    """Two tasks can land on one worker, so the pid alone is not unique."""
    paths = profile_job.write_flamegraphs(
        [summary(11, b"<html>a</html>"), summary(11, b"<html>b</html>")], tmp_path
    )

    assert len(set(paths)) == 2
    assert paths[0].read_bytes() != paths[1].read_bytes()


def test_parse_args_defaults():
    args = profile_job.parse_args([])

    assert args.tasks == profile_job.DEFAULT_TASKS
    assert args.batches == profile_job.DEFAULT_BATCHES
    assert args.index_depth == profile_job.DEFAULT_INDEX_DEPTH
    assert args.output_dir == profile_job.DEFAULT_OUTPUT_DIR
    assert args.leaks is False
    assert args.smoke_test is False


def test_parse_args_accepts_overrides():
    args = profile_job.parse_args(
        ["--tasks", "5", "--batches", "9", "--leaks", "--smoke-test"]
    )

    assert args.tasks == 5
    assert args.batches == 9
    assert args.leaks is True
    assert args.smoke_test is True


def test_main_launches_one_task_each_and_prints_a_report(stubbed_ray, tmp_path, capsys):
    calls, remote = stubbed_ray

    profile_job.main(["--tasks", "2", "--output-dir", str(tmp_path)])

    assert len(remote.calls) == 2
    assert calls["init"] is True
    assert calls["shutdown"] is True
    output = capsys.readouterr().out
    assert "101" in output
    assert "2 worker process(es)" in output


def test_main_writes_a_flamegraph_per_worker(stubbed_ray, tmp_path):
    _calls, _remote = stubbed_ray
    out = tmp_path / "reports"

    profile_job.main(["--tasks", "2", "--output-dir", str(out)])

    written = sorted(out.glob("*.html"))
    assert len(written) == 2
    assert {path.read_bytes() for path in written} == {
        b"<html>a</html>",
        b"<html>b</html>",
    }


def test_main_passes_the_workload_size_to_each_task(stubbed_ray, tmp_path):
    _calls, remote = stubbed_ray

    profile_job.main(
        [
            "--tasks",
            "1",
            "--batches",
            "7",
            "--frame-size",
            "128",
            "--index-depth",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert remote.calls[0] == (7, workload.DEFAULT_FRAMES_PER_BATCH, 128, 2, False)


def test_main_smoke_test_overrides_the_batch_count(stubbed_ray, tmp_path):
    _calls, remote = stubbed_ray

    profile_job.main(
        [
            "--tasks",
            "1",
            "--batches",
            "999",
            "--smoke-test",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert remote.calls[0][0] == profile_job.SMOKE_TEST_BATCHES
    assert remote.calls[0][3] == profile_job.SMOKE_TEST_INDEX_DEPTH


def test_main_forwards_the_leaks_flag(stubbed_ray, tmp_path):
    _calls, remote = stubbed_ray

    profile_job.main(["--tasks", "1", "--leaks", "--output-dir", str(tmp_path)])

    assert remote.calls[0][4] is True


def test_main_leaves_an_existing_ray_connection_alone(
    stubbed_ray, monkeypatch, tmp_path
):
    """Submitted as a Ray job, the driver connection is not ours to shut down."""
    calls, _remote = stubbed_ray
    monkeypatch.setattr(profile_job.ray, "is_initialized", lambda: True)

    profile_job.main(["--tasks", "1", "--output-dir", str(tmp_path)])

    assert "init" not in calls
    assert "shutdown" not in calls
