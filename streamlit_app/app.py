import boto3
import ray
import shutil
import socket
import streamlit as st
import time

from ray.job_submission import JobSubmissionClient, JobStatus

st.set_page_config(layout="wide")

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

def is_ray_running():
    try:
        s = socket.create_connection(("localhost", 8265), timeout=2)
        s.close()
        return True
    except:
        return False

# Conditional init if already running
if is_ray_running():
    ray.init(address='localhost:6379', ignore_reinit_error=True)

def wire_job(job_name: str, entrypoint: str, working_dir: str = "./streamlit_app/jobs"):
    if st.button(key=job_name, label=f"▶ Run {job_name} job"):
        client = JobSubmissionClient(address="http://localhost:8265")
        with st.spinner("Uploading code & submitting job…"):
            job_id = client.submit_job(
                entrypoint=entrypoint,
                runtime_env={"working_dir": working_dir},
            )
        st.success(f"{job_name} submitted: `{job_id}`")

        # Status and logs
        status_box = st.empty()
        log_box = st.empty()

        while True:
            status = client.get_job_status(job_id)
            status_box.info(f"Status: **{status}**")
            logs = client.get_job_logs(job_id)
            log_box.code(logs[-1024:], language="text")

            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                break
            time.sleep(5)

        if status == JobStatus.SUCCEEDED:
            st.success(f"✓ {job_name} finished successfully")
        else:
            st.error(f"{job_name} ended with status **{status}** — check logs above")

def check_minio_status():
    try:
        s = socket.create_connection(("localhost", 9000), timeout=2)
        s.close()
        return "✅ Running"
    except:
        return "❌ Not running"

def check_ray_status():
    return "✅ Running" if is_ray_running() else "❌ Not running"

def get_disk_usage():
    try:
        total, used, free = shutil.disk_usage('/')
        return f"{free // (1024**3)} GB free out of {total // (1024**3)} GB"
    except Exception as e:
        return f"Error: {e}"

st.title("ML Homelab Dashboard")

status_container = st.container()
with status_container:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📦 MinIO (Port 9000)")
        status = check_minio_status()
        st.markdown("✅ Online" if "✅" in status else "❌ Offline")
    with col2:
        st.markdown("### 🦊 Ray (Port 8265)")
        status = check_ray_status()
        st.markdown("✅ Online" if "✅" in status else "❌ Offline")

    with col3:
        st.markdown("### 📁 Disk Space")
        disk_usage = get_disk_usage()
        st.info(disk_usage)

tabs = st.tabs(["S3", "Training", "Inference"])

with tabs[0]:
    # Set up S3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )

    # List all available buckets
    try:
        buckets = s3.list_buckets()["Buckets"]
        bucket_names = [b["Name"] for b in buckets]
    except Exception as e:
        st.error(f"Failed to connect to S3/MinIO: {e}")
        bucket_names = []

    if bucket_names:
        selected_bucket = st.selectbox("Select a bucket to view contents", bucket_names)

        # List objects in the selected bucket
        try:
            objects = s3.list_objects_v2(Bucket=selected_bucket)
            object_list = objects.get("Contents", [])
            st.markdown(f"### 📂 Contents of `{selected_bucket}` ({len(object_list)} objects)")

            for obj in object_list:
                obj_key = obj["Key"]
                col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                with col1:
                    st.write(f"📄 `{obj_key}`")
                with col2:
                    size_kb = obj["Size"] / 1024
                    st.write(f"{size_kb:.1f} KB")
                with col3:
                    if st.button("⬇️", key=f"download-{obj_key}"):
                        url = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': selected_bucket, 'Key': obj_key},
                            ExpiresIn=3600
                        )
                        st.markdown(f"[Click to Download]({url})", unsafe_allow_html=True)
                with col4:
                    if st.button("🗑️", key=f"delete-{obj_key}"):
                        s3.delete_object(Bucket=selected_bucket, Key=obj_key)
                        st.warning(f"Deleted `{obj_key}`")
                        st.rerun()

                # Preview if supported filetype
                if obj_key.endswith((".txt", ".csv", ".json")):
                    body = s3.get_object(Bucket=selected_bucket, Key=obj_key)["Body"]
                    st.code(body.read().decode("utf-8")[:500], language="text")
                elif obj_key.endswith((".png", ".jpg", ".jpeg")):
                    img = s3.get_object(Bucket=selected_bucket, Key=obj_key)["Body"].read()
                    st.image(img, caption=obj_key, use_column_width=True)

        except Exception as e:
            st.error(f"Error listing objects in `{selected_bucket}`: {e}")

        st.markdown("---")

        # Upload a new file to the bucket
        st.markdown("### ⬆️ Upload a file to this bucket")
        with st.form("upload_form"):
            uploaded_file = st.file_uploader("Choose a file", type=None)
            if uploaded_file:
                dest_name = st.text_input("Object name in S3", value=uploaded_file.name)
            submitted = st.form_submit_button("Upload")

            if submitted and uploaded_file and dest_name:
                try:
                    s3.upload_fileobj(uploaded_file, selected_bucket, dest_name)
                    st.success(f"Uploaded `{dest_name}` to `{selected_bucket}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
    else:
        st.info("No buckets found. Please create one using MinIO Console or AWS CLI.")

with tabs[1]:
    training_jobs = [
        {
            "job_name": "MNIST Tune",
            "entrypoint": "python mnist_training/train_mnist.py"
        }
    ]

    for job in training_jobs:
        with st.expander(f"Training Job - {job['job_name']}", expanded=False):
            wire_job(job["job_name"], job["entrypoint"])

with tabs[2]:
    inference_jobs = [
        {
            "job_name": "Resnet Inference",
            "entrypoint": "python resnet_inference/inference.py"
        },
    ]

    for job in inference_jobs:
        with st.expander(f"Inference Job - {job['job_name']}", expanded=False):
            wire_job(job["job_name"], job["entrypoint"])