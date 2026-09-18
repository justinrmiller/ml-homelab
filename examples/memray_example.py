"""Profile a synthetic pipeline with memray and render a flamegraph.

Ray's metrics stack tells you *that* a process grew; memray tells you *which
Python code allocated it*. This example tracks the pipeline in
:mod:`streamlit_app.jobs.memray_profiling.workload` -- whose call tree is
shaped to make a flamegraph worth reading -- and renders the capture to HTML.

Run it with ``make memray``. See "Reading the flamegraph" in the README for a
walk through the result, and ``profile_job.py`` next to the workload for the
same tracker running inside Ray workers instead of locally.
"""

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import memray

from streamlit_app.jobs.memray_profiling import workload
from streamlit_app.storage import format_size

SMOKE_TEST_BATCHES = 8
SMOKE_TEST_INDEX_DEPTH = 3

DEFAULT_OUTPUT_DIR = ".memray"
CAPTURE_NAME = "profile.bin"
FLAMEGRAPH_NAME = "memray-flamegraph-profile.html"
LEAKS_NAME = "memray-leaks-profile.html"
TEMPORARY_NAME = "memray-temporary-profile.html"

# The three views memray can render over one capture. Each shows a different
# tier of the workload; see "Reading the flamegraph" in the README.
VIEWS = ("peak", "leaks", "temporary")

VIEW_FILENAMES: dict[str, str] = {
    "peak": FLAMEGRAPH_NAME,
    "leaks": LEAKS_NAME,
    "temporary": TEMPORARY_NAME,
}

VIEW_FLAGS: dict[str, list[str]] = {
    "peak": [],
    "leaks": ["--leaks"],
    "temporary": ["--temporary-allocations"],
}


def capture(
    capture_path: Path,
    batches: int = workload.DEFAULT_BATCHES,
    frames_per_batch: int = workload.DEFAULT_FRAMES_PER_BATCH,
    frame_size: int = workload.DEFAULT_FRAME_SIZE,
    index_depth: int = workload.DEFAULT_INDEX_DEPTH,
    trace_python_allocators: bool = False,
) -> Path:
    """Run the workload under a memray tracker and write the capture file.

    Args:
        capture_path: Destination for the ``.bin`` capture.
        batches: Number of batches to ingest.
        frames_per_batch: Records per batch.
        frame_size: Size of each record in bytes.
        index_depth: Levels of nesting in the index tree.
        trace_python_allocators: Record pymalloc's own allocations as well as
            the raw ``malloc`` calls underneath. Off by default, which is why
            small helpers served from an existing pool are invisible; see
            "Why some functions are missing" in the README.

    Returns:
        The path the capture was written to.
    """
    # The cache is global, so a second run would otherwise profile a workload
    # that starts out already full.
    workload.reset_cache()
    # overwrite=True because FileDestination refuses an existing file by
    # default, which would make the second `make memray` fail.
    destination = memray.FileDestination(str(capture_path), overwrite=True)
    with memray.Tracker(
        destination=destination, trace_python_allocators=trace_python_allocators
    ):
        workload.run_workload(batches, frames_per_batch, frame_size, index_depth)
    return capture_path


def summarize(capture_path: Path) -> dict[str, int]:
    """Read the headline numbers back out of a capture file.

    Args:
        capture_path: A ``.bin`` capture written by :func:`capture`.

    Returns:
        The capture's ``pid``, ``peak_memory`` and ``total_allocations``.
    """
    metadata = memray.FileReader(str(capture_path)).metadata
    return {
        "pid": metadata.pid,
        "peak_memory": metadata.peak_memory,
        "total_allocations": metadata.total_allocations,
    }


def format_summary(summary: dict[str, int]) -> str:
    """Render a capture summary as a human-readable block of text."""
    return "\n".join(
        [
            f"  pid:               {summary['pid']}",
            f"  peak memory:       {format_size(summary['peak_memory'])}",
            f"  total allocations: {summary['total_allocations']:,}",
        ]
    )


def flamegraph_command(
    capture_path: Path, html_path: Path, view: str = "peak"
) -> list[str]:
    """Build the ``memray flamegraph`` command line for one view.

    Invoked as ``python -m memray`` rather than the bare ``memray`` console
    script, which is not guaranteed to be on ``PATH``.

    Args:
        capture_path: The ``.bin`` capture to render.
        html_path: Destination HTML file.
        view: One of :data:`VIEWS`.

    Returns:
        The argument vector to execute.

    Raises:
        ValueError: If ``view`` is not one of :data:`VIEWS`.
    """
    if view not in VIEW_FLAGS:
        raise ValueError(f"unknown view {view!r}, expected one of {VIEWS}")

    return [
        sys.executable,
        "-m",
        "memray",
        "flamegraph",
        # Embed the report's assets instead of loading them from a CDN, so the
        # HTML opens on a machine with no internet access.
        "--no-web",
        "--force",
        "--output",
        str(html_path),
        *VIEW_FLAGS[view],
        str(capture_path),
    ]


def render_flamegraph(
    capture_path: Path,
    html_path: Path,
    view: str = "peak",
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    """Render a capture to an HTML flamegraph.

    Args:
        capture_path: The ``.bin`` capture to render.
        html_path: Destination HTML file.
        view: One of :data:`VIEWS`.
        runner: Subprocess runner, injectable so tests do not spawn memray.

    Returns:
        The path the HTML report was written to.
    """
    runner(flamegraph_command(capture_path, html_path, view), check=True)
    return html_path


def render_views(
    capture_path: Path,
    output_dir: Path,
    views: tuple[str, ...] = VIEWS,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Path]:
    """Render every requested view of one capture.

    Args:
        capture_path: The ``.bin`` capture to render.
        output_dir: Directory to write the HTML reports into.
        views: Which of :data:`VIEWS` to render.
        runner: Subprocess runner, injectable so tests do not spawn memray.

    Returns:
        A mapping of view name to the HTML file written for it.
    """
    return {
        view: render_flamegraph(
            capture_path, output_dir / VIEW_FILENAMES[view], view, runner
        )
        for view in views
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the profiling run."""
    parser = argparse.ArgumentParser(
        description="Profile a synthetic allocation workload with memray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the capture file and the HTML reports",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=workload.DEFAULT_BATCHES,
        help="Batches to ingest",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=workload.DEFAULT_FRAMES_PER_BATCH,
        help="Records retained per batch",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=workload.DEFAULT_FRAME_SIZE,
        help="Size of each record, in bytes",
    )
    parser.add_argument(
        "--index-depth",
        type=int,
        default=workload.DEFAULT_INDEX_DEPTH,
        help="Levels of recursion in the index tree",
    )
    parser.add_argument(
        "--view",
        choices=[*VIEWS, "all"],
        default="all",
        help="Which flamegraph view(s) to render",
    )
    parser.add_argument(
        "--trace-python-allocators",
        action="store_true",
        help="Also record pymalloc's allocations, revealing small helpers",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Ingest only {SMOKE_TEST_BATCHES} batches, to finish quickly",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Capture a profile of the synthetic pipeline and render its views."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / CAPTURE_NAME

    batches = SMOKE_TEST_BATCHES if args.smoke_test else args.batches
    index_depth = SMOKE_TEST_INDEX_DEPTH if args.smoke_test else args.index_depth

    print(f"Tracking {batches} batches into {capture_path} …")
    capture(
        capture_path,
        batches,
        args.frames_per_batch,
        args.frame_size,
        index_depth,
        args.trace_python_allocators,
    )

    print(format_summary(summarize(capture_path)))

    views = VIEWS if args.view == "all" else (args.view,)
    print()
    for view, path in render_views(capture_path, output_dir, views).items():
        print(f"{view:>10} view: {path}")


if __name__ == "__main__":
    main()
