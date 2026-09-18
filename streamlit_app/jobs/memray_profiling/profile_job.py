"""Profile Ray workers with memray.

``memray.Tracker`` profiles the process it runs in. Wrapping the *driver* would
only measure the process that hands work out, so the tracker here is opened
inside the Ray task instead: each task profiles the worker executing it.

Getting the result somewhere you can look at it is the other half of the
problem. Each task renders its own capture to a self-contained HTML flamegraph
inside the worker and returns the *bytes* over the object store; the driver
writes them out, one file per worker, alongside a summary table in the job log.
Where those files land depends on how the job runs:

* Against a local Ray (``python profile_job.py``), the driver is on your
  machine, so ``--output-dir`` is a local directory you can open directly.
* Submitted to KubeRay, the driver runs in a pod, so copy them out with
  ``kubectl cp``. Ray's own dashboard is usually easier -- see "Viewing Ray
  worker profiles" in the README.

The workload lives in ``workload.py`` next door, shared with
``examples/memray_example.py`` so the two flamegraphs are comparable.
"""

import argparse
import os
import tempfile
import uuid
from pathlib import Path
from typing import TypedDict

import memray
import ray

try:
    # Imported from the repository: by the tests, ruff, and ty.
    from streamlit_app.jobs.memray_profiling import workload
except ImportError:  # pragma: no cover - only taken on the cluster
    # Submitted as a Ray job: only `streamlit_app/jobs` is uploaded, and Python
    # puts this script's own directory on sys.path, so the sibling resolves.
    import workload  # ty: ignore[unresolved-import]

# Smaller than the local example's defaults: Ray workers are capped at 3Gi and
# several tasks profile at once.
DEFAULT_TASKS = 2
DEFAULT_BATCHES = 256
DEFAULT_INDEX_DEPTH = 5
SMOKE_TEST_BATCHES = 8
SMOKE_TEST_INDEX_DEPTH = 3

DEFAULT_OUTPUT_DIR = ".memray-ray"

MIB = 1024**2


class WorkerProfile(TypedDict):
    """What one task reports back about the worker that ran it.

    Attributes:
        pid: The worker process the tracker ran in.
        peak_memory: High watermark, in bytes.
        total_allocations: Allocations memray recorded.
        retained_bytes: What the workload deliberately held on to.
        flamegraph: The rendered HTML report, self-contained.
    """

    pid: int
    peak_memory: int
    total_allocations: int
    retained_bytes: int
    flamegraph: bytes


def capture_path(directory: str | None = None) -> str:
    """Build a capture path unique to this process and call.

    Ray reuses worker processes across tasks, so a fixed filename would collide
    with the capture written by whichever task ran here before.

    Args:
        directory: Directory to place the capture in. Defaults to the system
            temporary directory.

    Returns:
        An absolute path that no other task will pick.
    """
    base = directory or tempfile.gettempdir()
    return os.path.join(base, f"memray-{os.getpid()}-{uuid.uuid4().hex}.bin")


def flamegraph_bytes(capture_file: str, leaks: bool = False) -> bytes:
    """Render a capture to HTML inside the worker and read it back.

    Shelling out keeps this honest: ``memray flamegraph`` is the only supported
    way to produce the report, and there is no public Python reporter API.

    Args:
        capture_file: The ``.bin`` capture to render.
        leaks: Render the memory-leaks view instead of peak memory.

    Returns:
        The rendered HTML, self-contained so it opens without network access.
    """
    # Imported here rather than at module scope: the driver never renders, so
    # only the workers pay for it.
    import subprocess
    import sys

    html_file = f"{capture_file}.html"
    command = [
        sys.executable,
        "-m",
        "memray",
        "flamegraph",
        "--no-web",
        "--force",
        "--output",
        html_file,
    ]
    if leaks:
        command.append("--leaks")
    command.append(capture_file)

    subprocess.run(command, check=True, capture_output=True)
    return Path(html_file).read_bytes()


def profile_task(
    batches: int,
    frames_per_batch: int,
    frame_size: int,
    index_depth: int,
    leaks: bool,
    directory: str | None = None,
) -> WorkerProfile:
    """Run the workload under a tracker and report what the worker allocated.

    Kept undecorated so it is callable directly in tests; ``profile_remote``
    below is the handle Ray schedules on the cluster.

    Every argument :func:`main` passes through Ray is required rather than
    defaulted. ``ray.remote`` is typed with one overload per arity, and it
    resolves against the *required* parameters only -- a default here silently
    narrows the handle and makes the call in :func:`main` a type error.
    ``directory`` keeps its default because Ray never passes it.

    Args:
        batches: Number of batches to ingest.
        frames_per_batch: Records per batch.
        frame_size: Size of each record in bytes.
        index_depth: Levels of nesting in the index tree.
        leaks: Render the memory-leaks view instead of peak memory.
        directory: Directory for the capture file.

    Returns:
        The worker's ``pid``, ``peak_memory``, ``total_allocations``,
        ``retained_bytes``, and the rendered ``flamegraph`` HTML bytes.
    """
    # The cache outlives the task in a reused worker, so clear it first or the
    # second task on this worker profiles a head start it did not allocate.
    workload.reset_cache()

    path = capture_path(directory)
    destination = memray.FileDestination(path, overwrite=True)
    with memray.Tracker(destination=destination):
        retained, _transient = workload.run_workload(
            batches, frames_per_batch, frame_size, index_depth
        )

    metadata = memray.FileReader(path).metadata
    return WorkerProfile(
        pid=metadata.pid,
        peak_memory=metadata.peak_memory,
        total_allocations=metadata.total_allocations,
        retained_bytes=retained,
        flamegraph=flamegraph_bytes(path, leaks),
    )


# The remote handle Ray schedules on the cluster.
profile_remote = ray.remote(profile_task)


def format_report(summaries: list[WorkerProfile]) -> str:
    """Render one row per profiled worker.

    Args:
        summaries: Summaries returned by :func:`profile_task`.

    Returns:
        A table with a header row, ready to print.
    """
    header = f"{'pid':>8}  {'peak':>12}  {'allocations':>13}  {'retained':>12}"
    rows = [
        f"{summary['pid']:>8}  "
        f"{summary['peak_memory'] / MIB:>9.1f} MB  "
        f"{summary['total_allocations']:>13,}  "
        f"{summary['retained_bytes'] / MIB:>9.1f} MB"
        for summary in summaries
    ]
    return "\n".join([header, "-" * len(header), *rows])


def distinct_pids(summaries: list[WorkerProfile]) -> int:
    """Count how many separate worker processes the summaries came from."""
    return len({summary["pid"] for summary in summaries})


def write_flamegraphs(summaries: list[WorkerProfile], output_dir: Path) -> list[Path]:
    """Write each worker's flamegraph HTML to ``output_dir``.

    Several tasks can land on one worker, so the filename carries the task's
    index as well as the pid.

    Args:
        summaries: Summaries returned by :func:`profile_task`.
        output_dir: Directory to write into; created if absent.

    Returns:
        The paths written, in the order the summaries were given.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, summary in enumerate(summaries):
        path = output_dir / f"memray-worker-{summary['pid']}-task{index}.html"
        path.write_bytes(summary["flamegraph"])
        paths.append(path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the profiling job."""
    parser = argparse.ArgumentParser(
        description="Profile Ray workers with memray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks", type=int, default=DEFAULT_TASKS, help="Tasks to launch"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=DEFAULT_BATCHES,
        help="Batches each task ingests",
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
        default=DEFAULT_INDEX_DEPTH,
        help="Levels of recursion in the index tree",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the driver writes each worker's flamegraph into",
    )
    parser.add_argument(
        "--leaks",
        action="store_true",
        help="Render the memory-leaks view instead of peak memory",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Ingest only {SMOKE_TEST_BATCHES} batches per task",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Profile several Ray workers and write a flamegraph for each."""
    args = parse_args(argv)
    batches = SMOKE_TEST_BATCHES if args.smoke_test else args.batches
    index_depth = SMOKE_TEST_INDEX_DEPTH if args.smoke_test else args.index_depth

    # Only tear Ray down at the end if we were the ones who started it: when
    # this runs as a submitted Ray job, the driver connection is not ours.
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init()

    try:
        print(f"Profiling {args.tasks} task(s), {batches} batches each …")
        summaries = ray.get(
            [
                profile_remote.remote(
                    batches,
                    args.frames_per_batch,
                    args.frame_size,
                    index_depth,
                    args.leaks,
                )
                for _ in range(args.tasks)
            ]
        )

        print(format_report(summaries))
        # The tracker ran in the workers, not the driver. Distinct pids are the
        # evidence; one shared pid just means Ray reused a single worker.
        print(
            f"\nProfiled {len(summaries)} task(s) across "
            f"{distinct_pids(summaries)} worker process(es)."
        )

        print()
        for path in write_flamegraphs(summaries, Path(args.output_dir)):
            print(f"  flamegraph: {path}")
    finally:
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    main()
