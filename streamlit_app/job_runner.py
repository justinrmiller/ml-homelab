"""Ray job submission and polling helpers.

Jobs are submitted over the Ray Jobs HTTP API rather than the Ray Client
protocol (``ray://``): the client protocol spawns a proxy server per connection
and requires the client and cluster Python versions to match.
"""

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import yaml
from ray.job_submission import JobStatus, JobSubmissionClient

RAY_DASHBOARD_URL = "http://localhost:8265"
DEFAULT_WORKING_DIR = "./streamlit_app/jobs"
# CPUs reserved for the job's driver process. This is not really about CPU: a
# submission that asks for no resources gets a head-node label selector, which
# pins the driver to the head. Ray 2.58 runs each dashboard module as its own
# process, which already claims most of the head pod's memory, so a driver that
# imports torch there OOM-kills the node. Any non-zero request drops the selector, and since the head advertises
# 0 CPUs the driver lands on a worker instead. The trade-off is that a job
# submitted to a cluster whose workers are not up yet waits for one instead of
# running on the head.
ENTRYPOINT_NUM_CPUS = 1
DEFAULT_POLL_INTERVAL = 5.0
LOG_TAIL_CHARS = 1024

TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
)


@dataclass(frozen=True)
class JobSpec:
    """A job the dashboard knows how to submit.

    Attributes:
        name: Display name, also used as the Streamlit widget key.
        entrypoint: Command run inside the job's runtime environment.
        runtime_env_file: Optional runtime env YAML, relative to the working dir.
    """

    name: str
    entrypoint: str
    runtime_env_file: str | None = None


TRAINING_JOBS: tuple[JobSpec, ...] = (
    JobSpec(
        name="MNIST Tune",
        entrypoint="python mnist_training/train_mnist.py",
        runtime_env_file="mnist_training/runtime_env.yaml",
    ),
)

INFERENCE_JOBS: tuple[JobSpec, ...] = (
    JobSpec(
        name="Resnet Inference",
        entrypoint="python resnet_inference/inference.py",
        runtime_env_file="resnet_inference/runtime_env.yaml",
    ),
)

PROFILING_JOBS: tuple[JobSpec, ...] = (
    JobSpec(
        name="Memray Profile",
        entrypoint="python memray_profiling/profile_job.py",
        runtime_env_file="memray_profiling/runtime_env.yaml",
    ),
)


def load_runtime_env(
    working_dir: str = DEFAULT_WORKING_DIR, runtime_env_file: str | None = None
) -> dict[str, Any]:
    """Build a Ray runtime environment, merging in a YAML file when present.

    A missing or empty YAML file is not an error: the working directory alone is
    a valid runtime environment.

    Args:
        working_dir: Directory uploaded to the cluster with the job.
        runtime_env_file: Optional YAML path relative to ``working_dir``.

    Returns:
        The runtime environment dict to hand to the Jobs API.
    """
    runtime_env: dict[str, Any] = {"working_dir": working_dir}
    if not runtime_env_file:
        return runtime_env

    env_path = os.path.join(working_dir, runtime_env_file)
    if not os.path.isfile(env_path):
        return runtime_env

    with open(env_path) as handle:
        extra = yaml.safe_load(handle) or {}
    runtime_env.update(extra)
    return runtime_env


def create_client(address: str = RAY_DASHBOARD_URL) -> JobSubmissionClient:
    """Create a Jobs API client pointed at the Ray dashboard."""
    return JobSubmissionClient(address=address)


def submit_job(
    client: Any, spec: JobSpec, working_dir: str = DEFAULT_WORKING_DIR
) -> str:
    """Submit ``spec`` to the cluster and return the resulting job id."""
    return client.submit_job(
        entrypoint=spec.entrypoint,
        runtime_env=load_runtime_env(working_dir, spec.runtime_env_file),
        entrypoint_num_cpus=ENTRYPOINT_NUM_CPUS,
    )


def is_terminal(status: JobStatus) -> bool:
    """Report whether a job status means the job has stopped running."""
    return status in TERMINAL_STATUSES


def tail_logs(logs: str, limit: int = LOG_TAIL_CHARS) -> str:
    """Return the last ``limit`` characters of a log blob."""
    return logs[-limit:]


def poll_job(
    client: Any,
    job_id: str,
    on_update: Callable[[JobStatus, str], None] | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    sleep: Callable[[float], None] = time.sleep,
) -> JobStatus:
    """Poll a job until it reaches a terminal status.

    Args:
        client: A Jobs API client.
        job_id: Id returned by :func:`submit_job`.
        on_update: Called with ``(status, log_tail)`` after every poll.
        poll_interval: Seconds to wait between polls.
        sleep: Sleep function, injectable so tests do not wait.

    Returns:
        The terminal :class:`JobStatus`.
    """
    while True:
        status = client.get_job_status(job_id)
        if on_update is not None:
            on_update(status, tail_logs(client.get_job_logs(job_id)))
        if is_terminal(status):
            return status
        sleep(poll_interval)


def summarize_result(name: str, status: JobStatus) -> tuple[bool, str]:
    """Build the success flag and message shown when a job finishes.

    Args:
        name: Display name of the job.
        status: Terminal status of the job.

    Returns:
        ``(succeeded, message)``.
    """
    if status == JobStatus.SUCCEEDED:
        return True, f"✓ {name} finished successfully"
    return False, f"{name} ended with status **{status}** — check logs above"


def job_names(specs: Iterable[JobSpec]) -> list[str]:
    """Return the display names for a collection of job specs."""
    return [spec.name for spec in specs]
