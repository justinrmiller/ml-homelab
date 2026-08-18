"""Ray Job Submission Example.

Submits ``examples/hello_ray_job.py`` through the Ray Jobs HTTP API and tails
its logs. Run it from the repository root as a module so the ``streamlit_app``
package resolves:

    uv run python -m examples.ray_job_example
"""

from streamlit_app.job_runner import (
    RAY_DASHBOARD_URL,
    JobSpec,
    create_client,
    poll_job,
    submit_job,
)

HELLO_JOB = JobSpec(name="hello", entrypoint="python examples/hello_ray_job.py")


def main(address: str = RAY_DASHBOARD_URL) -> None:
    """Submit the hello job, poll it to completion, and print its logs."""
    client = create_client(address)
    job_id = submit_job(client, HELLO_JOB, working_dir="./")
    print(f"✅ Job submitted: {job_id}")

    def report(status, log_tail: str) -> None:
        print(f"Status: {status}")

    poll_job(client, job_id, on_update=report, poll_interval=1.0)

    print("\n📄 Job Logs:\n", client.get_job_logs(job_id))


if __name__ == "__main__":
    main()
