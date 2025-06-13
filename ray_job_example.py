# ray_job_single_file.py

import ray
from ray.job_submission import JobSubmissionClient, JobStatus
import time
import os

# 2. Submit job using JobSubmissionClient
client = JobSubmissionClient("ray://127.0.0.1:10001")
job_id = client.submit_job(
    entrypoint=f"python hello_ray_job.py",
    runtime_env={"working_dir": "./"},
)
print(f"✅ Job submitted: {job_id}")

# 3. Poll until job completes
while True:
    status = client.get_job_status(job_id)
    print(f"Status: {status}")
    if status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED]:
        break
    time.sleep(1)

# 4. Fetch logs
logs = client.get_job_logs(job_id)
print("\n📄 Job Logs:\n", logs)
