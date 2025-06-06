import streamlit as st
import ray
import boto3
import pandas as pd
import time                                         # NEW
from ray.job_submission import JobSubmissionClient  # NEW
from ray.job_submission import JobStatus            # NEW

st.set_page_config(layout="wide")

st.title("ML Homelab Dashboard")

ray.init(address='auto', ignore_reinit_error=True)

if st.button("List S3 Buckets"):
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )
    try:
        buckets = s3.list_buckets()
        for bucket in buckets["Buckets"]:
            st.write(f"Bucket: {bucket['Name']}")
    except Exception as e:
        st.error(f"Failed to list buckets: {e}")

st.subheader("Training Jobs")

with st.expander("Training Jobs"):
    if st.button("▶ Run MNIST Tune job"):
        client = JobSubmissionClient(address="http://localhost:8265")

        with st.spinner("Uploading code & submitting job…"):
            job_id = client.submit_job(
                entrypoint="python mnist_training/train_mnist.py",
                runtime_env={"working_dir": "./streamlit_app/jobs"},
            )
        st.success(f"Job submitted: `{job_id}`")

        # Placeholders so the layout doesn’t jump around
        status_box = st.empty()
        log_box = st.empty()

        # Poll every few seconds until the job reaches a terminal state
        while True:
            status = client.get_job_status(job_id)
            status_box.info(f"Status: **{status}**")

            # Stream the latest logs (truncate to last ~1 kB for readability)
            logs = client.get_job_logs(job_id)
            log_box.code(logs[-1024:], language="text")

            if status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.STOPPED,
            ):
                break
            time.sleep(5)

        # Final user feedback
        if status == JobStatus.SUCCEEDED:
            st.success("✓ Training finished successfully")
        else:
            st.error(f"Job ended with status **{status}** — check logs above")

st.subheader("Inference Jobs")

with st.expander("Inference Jobs"):
    if st.button("▶ Run ResNet Inference job"):
        client = JobSubmissionClient(address="http://localhost:8265")

        with st.spinner("Uploading code & submitting job…"):
            job_id = client.submit_job(
                entrypoint="python resnet_inference/inference.py",
                runtime_env={"working_dir": "./streamlit_app/jobs"},
            )
        st.success(f"Job submitted: `{job_id}`")

        # Placeholders so the layout doesn’t jump around
        status_box = st.empty()
        log_box = st.empty()

        # Poll every few seconds until the job reaches a terminal state
        while True:
            status = client.get_job_status(job_id)
            status_box.info(f"Status: **{status}**")

            # Stream the latest logs (truncate to last ~1 kB for readability)
            logs = client.get_job_logs(job_id)
            log_box.code(logs[-1024:], language="text")

            if status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.STOPPED,
            ):
                break
            time.sleep(5)

        # Final user feedback
        if status == JobStatus.SUCCEEDED:
            st.success("✓ ResNet inference finished successfully")
        else:
            st.error(f"Job ended with status **{status}** — check logs above")
