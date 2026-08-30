"""Streamlit dashboard for the ML homelab."""

import base64
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
# Prevent Ray auto-connecting via Ray Client protocol (ray://) which requires
# matching Python versions between client and cluster. JobSubmissionClient uses
# HTTP instead, so RAY_ADDRESS is not needed here. This has to happen before
# anything imports ray.
os.environ.pop("RAY_ADDRESS", None)

# `streamlit run streamlit_app/app.py` puts this file's own directory on
# sys.path, not the repository root, so the absolute `streamlit_app` imports
# below do not resolve on their own. pytest supplies the root via
# `pythonpath = ["."]` in pyproject.toml, which is why the tests pass either
# way; the app has to arrange it for itself. Must precede the first
# `streamlit_app` import.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from streamlit_app import health, storage  # noqa: E402
from streamlit_app.job_runner import (  # noqa: E402
    INFERENCE_JOBS,
    TRAINING_JOBS,
    JobSpec,
    create_client,
    poll_job,
    submit_job,
    summarize_result,
)

st.set_page_config(layout="wide")

CSS = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# The Ray logo is an SVG rather than an emoji, so it rides into the heading as a
# base64 data URI: Streamlit sanitizes inline <svg> out of markdown, but leaves
# an <img> with a data source alone.
_RAY_LOGO = base64.b64encode(
    (Path(__file__).parent / "assets" / "ray-logo.svg").read_bytes()
).decode()
RAY_LOGO_IMG = (
    f'<img src="data:image/svg+xml;base64,{_RAY_LOGO}" alt="Ray"'
    ' style="height:1em;width:1em;vertical-align:-0.12em;">'
)


def render_job(spec: JobSpec) -> None:
    """Render the run button for a job and stream its progress once clicked."""
    if not st.button(key=spec.name, label=f"▶ Run {spec.name} job"):
        return

    client = create_client()
    with st.spinner("Uploading code & submitting job…"):
        job_id = submit_job(client, spec)
    st.success(f"{spec.name} submitted: `{job_id}`")

    status_box = st.empty()
    log_box = st.empty()

    def on_update(status, log_tail: str) -> None:
        status_box.info(f"Status: **{status}**")
        log_box.code(log_tail, language="text")

    status = poll_job(client, job_id, on_update=on_update)

    succeeded, message = summarize_result(spec.name, status)
    (st.success if succeeded else st.error)(message)


def render_status_header() -> None:
    """Render the service status and disk usage row."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### 📦 Floci S3 (Port {health.FLOCI_PORT})")
        st.markdown(health.status_label(health.is_floci_running()))
    with col2:
        st.markdown(
            f"<h3>{RAY_LOGO_IMG} Ray (Port {health.RAY_DASHBOARD_PORT})</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(health.status_label(health.is_ray_running()))
    with col3:
        st.markdown("### 📁 Disk Space")
        st.info(health.get_disk_usage())


def render_object_row(client: Any, bucket: str, obj: dict) -> None:
    """Render one object's row: name, size, download, delete, and preview."""
    key = obj["Key"]
    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
    with col1:
        st.write(f"📄 `{key}`")
    with col2:
        st.write(storage.format_size(obj["Size"]))
    with col3:
        if st.button("⬇️", key=f"download-{key}"):
            url = storage.presigned_download_url(client, bucket, key)
            st.markdown(f"[Click to Download]({url})", unsafe_allow_html=True)
    with col4:
        if st.button("🗑️", key=f"delete-{key}"):
            storage.delete_object(client, bucket, key)
            st.warning(f"Deleted `{key}`")
            st.rerun()

    render_preview(client, bucket, key)


def render_preview(client: Any, bucket: str, key: str) -> None:
    """Render an inline preview of an object, if its type supports one."""
    kind = storage.preview_kind(key)
    if kind is None:
        return

    try:
        if kind == "text":
            st.code(storage.read_text_preview(client, bucket, key), language="text")
        else:
            st.image(
                storage.read_object_bytes(client, bucket, key),
                caption=key,
                width="stretch",
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.warning(f"Could not preview `{key}`: {exc}")


def render_upload_form(client: Any, bucket: str) -> None:
    """Render the upload form for the selected bucket.

    The name field is rendered unconditionally: widgets inside a form do not
    rerun the script, so one that only appears once a file is attached would
    never be editable before the upload it is meant to name.

    The submit branch is the one part of the UI AppTest cannot drive, since it
    has no way to attach a file to ``st.file_uploader``; the logic behind it is
    covered by :func:`streamlit_app.storage.resolve_object_key` and
    :func:`streamlit_app.storage.try_upload`.
    """
    st.markdown("### ⬆️ Upload a file to this bucket")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a file", type=None)
        dest_name = st.text_input(
            "Object name in S3", placeholder="defaults to the name of the file"
        )
        submitted = st.form_submit_button("Upload")

        if not (submitted and uploaded_file):
            return

        key = storage.resolve_object_key(dest_name, uploaded_file.name)
        succeeded, message = storage.try_upload(client, uploaded_file, bucket, key)
        if not succeeded:
            st.error(message)
            return
        st.success(message)
        st.rerun()


def render_storage_tab() -> None:
    """Render the Floci bucket browser."""
    client = storage.create_client()

    try:
        bucket_names = storage.list_bucket_names(client)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"Failed to connect to S3 (Floci): {exc}")
        bucket_names = []

    if not bucket_names:
        st.info(
            "No buckets found. Create one with the AWS CLI: "
            f"`aws s3 mb s3://my-bucket --endpoint-url {storage.DEFAULT_ENDPOINT_URL}`."
        )
        return

    bucket = st.selectbox("Select a bucket to view contents", bucket_names)

    try:
        objects = storage.list_objects(client, bucket)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"Error listing objects in `{bucket}`: {exc}")
        objects = []
    else:
        st.markdown(f"### 📂 Contents of `{bucket}` ({len(objects)} objects)")

    for obj in objects:
        render_object_row(client, bucket, obj)

    st.markdown("---")
    render_upload_form(client, bucket)


def render_job_tab(specs: Iterable[JobSpec], label: str) -> None:
    """Render an expander with a run button for each job spec."""
    for spec in specs:
        with st.expander(f"{label} - {spec.name}", expanded=False):
            render_job(spec)


st.title("ML Homelab Dashboard")

with st.container():
    render_status_header()

tabs = st.tabs(["S3", "Training", "Inference"])

with tabs[0]:
    render_storage_tab()

with tabs[1]:
    render_job_tab(TRAINING_JOBS, "Training Job")

with tabs[2]:
    render_job_tab(INFERENCE_JOBS, "Inference Job")
