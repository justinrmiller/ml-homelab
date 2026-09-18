"""Tests for the local memray profiling example."""

import sys

import memray
import pytest

from examples import memray_example
from streamlit_app.jobs.memray_profiling import workload


class FakeRunner:
    """Records the command lines it is handed instead of spawning memray."""

    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return None


class FakeRenderer:
    """Stands in for render_views so main() never spawns memray.

    render_flamegraph binds subprocess.run as a default argument at import
    time, so monkeypatching the subprocess module would not reach it. main()
    looks this up on the module at call time, which does.
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, capture_path, output_dir, views=memray_example.VIEWS):
        self.calls.append((capture_path, output_dir, tuple(views)))
        return {view: output_dir / f"{view}.html" for view in views}


@pytest.fixture(autouse=True)
def clean_cache():
    workload.reset_cache()
    yield
    workload.reset_cache()


@pytest.fixture
def runner():
    return FakeRunner()


def test_capture_writes_a_readable_file(tmp_path):
    path = tmp_path / "profile.bin"

    returned = memray_example.capture(
        path, batches=2, frames_per_batch=4, frame_size=1024, index_depth=1
    )

    assert returned == path
    assert path.is_file()
    assert memray.FileReader(str(path)).metadata.pid > 0


def test_capture_overwrites_an_existing_file(tmp_path):
    """FileDestination refuses an existing file unless overwrite is set."""
    path = tmp_path / "profile.bin"
    memray_example.capture(
        path, batches=1, frames_per_batch=2, frame_size=512, index_depth=1
    )

    memray_example.capture(
        path, batches=1, frames_per_batch=2, frame_size=512, index_depth=1
    )

    assert path.is_file()


def test_capture_resets_the_cache_first(tmp_path):
    workload.ingest_batches(batches=16, frames_per_batch=2, frame_size=8)

    memray_example.capture(
        tmp_path / "profile.bin",
        batches=2,
        frames_per_batch=2,
        frame_size=8,
        index_depth=1,
    )

    assert len([key for key in workload.CACHE if key.startswith("batch-")]) == 2


def test_summarize_reports_real_numbers(tmp_path):
    path = tmp_path / "profile.bin"
    memray_example.capture(
        path, batches=8, frames_per_batch=8, frame_size=4096, index_depth=2
    )

    summary = memray_example.summarize(path)

    assert set(summary) == {"pid", "peak_memory", "total_allocations"}
    assert summary["peak_memory"] > 0
    assert summary["total_allocations"] > 0


def test_format_summary_renders_every_field():
    text = memray_example.format_summary(
        {"pid": 4321, "peak_memory": 2 * 1024**2, "total_allocations": 1234}
    )

    assert "4321" in text
    assert "2.0 MB" in text
    assert "1,234" in text


def test_flamegraph_command_targets_the_module_not_the_console_script(tmp_path):
    command = memray_example.flamegraph_command(
        tmp_path / "profile.bin", tmp_path / "out.html"
    )

    assert command[:4] == [sys.executable, "-m", "memray", "flamegraph"]
    assert command[-1] == str(tmp_path / "profile.bin")
    assert "--force" in command
    assert "--no-web" in command
    assert "--leaks" not in command


@pytest.mark.parametrize(
    ("view", "flag"),
    [("leaks", "--leaks"), ("temporary", "--temporary-allocations")],
)
def test_flamegraph_command_adds_the_view_flag(tmp_path, view, flag):
    command = memray_example.flamegraph_command(
        tmp_path / "profile.bin", tmp_path / "out.html", view
    )

    assert flag in command


def test_flamegraph_command_rejects_an_unknown_view(tmp_path):
    with pytest.raises(ValueError, match="unknown view"):
        memray_example.flamegraph_command(
            tmp_path / "profile.bin", tmp_path / "out.html", "nonsense"
        )


def test_every_view_has_a_filename_and_flags():
    assert set(memray_example.VIEW_FILENAMES) == set(memray_example.VIEWS)
    assert set(memray_example.VIEW_FLAGS) == set(memray_example.VIEWS)
    assert len(set(memray_example.VIEW_FILENAMES.values())) == len(memray_example.VIEWS)


def test_render_flamegraph_runs_the_command(tmp_path, runner):
    html = tmp_path / "out.html"

    returned = memray_example.render_flamegraph(
        tmp_path / "profile.bin", html, runner=runner
    )

    assert returned == html
    command, kwargs = runner.calls[0]
    assert command == memray_example.flamegraph_command(tmp_path / "profile.bin", html)
    assert kwargs == {"check": True}


def test_render_views_renders_each_requested_view(tmp_path, runner):
    paths = memray_example.render_views(
        tmp_path / "profile.bin", tmp_path, runner=runner
    )

    assert set(paths) == set(memray_example.VIEWS)
    assert len(runner.calls) == len(memray_example.VIEWS)
    assert len(set(paths.values())) == len(memray_example.VIEWS)


def test_render_views_can_render_a_subset(tmp_path, runner):
    paths = memray_example.render_views(
        tmp_path / "profile.bin", tmp_path, views=("leaks",), runner=runner
    )

    assert set(paths) == {"leaks"}
    assert "--leaks" in runner.calls[0][0]


def test_parse_args_defaults():
    args = memray_example.parse_args([])

    assert args.output_dir == memray_example.DEFAULT_OUTPUT_DIR
    assert args.batches == workload.DEFAULT_BATCHES
    assert args.frames_per_batch == workload.DEFAULT_FRAMES_PER_BATCH
    assert args.frame_size == workload.DEFAULT_FRAME_SIZE
    assert args.index_depth == workload.DEFAULT_INDEX_DEPTH
    assert args.view == "all"
    assert args.smoke_test is False


def test_parse_args_accepts_overrides():
    args = memray_example.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--batches",
            "3",
            "--view",
            "leaks",
            "--smoke-test",
        ]
    )

    assert args.output_dir == "/tmp/out"
    assert args.batches == 3
    assert args.view == "leaks"
    assert args.smoke_test is True


def test_main_creates_the_output_dir_and_renders_every_view(
    tmp_path, monkeypatch, capsys
):
    renderer = FakeRenderer()
    monkeypatch.setattr(memray_example, "render_views", renderer)
    out = tmp_path / "reports"

    memray_example.main(
        ["--output-dir", str(out), "--smoke-test", "--frame-size", "1024"]
    )

    assert (out / memray_example.CAPTURE_NAME).is_file()
    assert renderer.calls == [
        (out / memray_example.CAPTURE_NAME, out, memray_example.VIEWS)
    ]
    assert "peak memory" in capsys.readouterr().out


def test_main_can_render_a_single_view(tmp_path, monkeypatch):
    renderer = FakeRenderer()
    monkeypatch.setattr(memray_example, "render_views", renderer)

    memray_example.main(
        [
            "--output-dir",
            str(tmp_path),
            "--smoke-test",
            "--frame-size",
            "512",
            "--view",
            "leaks",
        ]
    )

    assert renderer.calls[0][2] == ("leaks",)


def test_main_smoke_test_shrinks_the_workload(tmp_path, monkeypatch):
    monkeypatch.setattr(memray_example, "render_views", FakeRenderer())

    memray_example.main(
        ["--output-dir", str(tmp_path), "--smoke-test", "--frame-size", "512"]
    )

    batches = [key for key in workload.CACHE if key.startswith("batch-")]
    assert len(batches) == memray_example.SMOKE_TEST_BATCHES


def test_capture_can_trace_python_allocators(tmp_path):
    """Pymalloc tracing records far more allocations than the default."""
    plain = memray_example.capture(
        tmp_path / "plain.bin",
        batches=4,
        frames_per_batch=8,
        frame_size=1024,
        index_depth=1,
    )
    traced = memray_example.capture(
        tmp_path / "traced.bin",
        batches=4,
        frames_per_batch=8,
        frame_size=1024,
        index_depth=1,
        trace_python_allocators=True,
    )

    assert (
        memray_example.summarize(traced)["total_allocations"]
        > memray_example.summarize(plain)["total_allocations"]
    )


def test_parse_args_defaults_to_no_python_allocator_tracing():
    assert memray_example.parse_args([]).trace_python_allocators is False
