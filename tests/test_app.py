"""End-to-end tests of the Streamlit dashboard using Streamlit's AppTest runner.

The app script is executed headlessly with the S3, Ray, and health-check layers
replaced by the in-memory doubles from ``conftest``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from ray.job_submission import JobStatus
from streamlit.testing.v1 import AppTest

from streamlit_app import health, job_runner, storage
from tests.conftest import FakeJobClient, FakeS3Client

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app" / "app.py")
APP_TIMEOUT = 30


@pytest.fixture
def app(monkeypatch, s3_client):
    """An AppTest wired to fakes, with both services reported as up."""
    monkeypatch.setattr(storage, "create_client", lambda: s3_client)
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: True)
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT)


def test_app_renders_without_exceptions(app):
    app.run()
    assert not app.exception


def test_app_renders_title_and_tabs(app):
    app.run()
    assert "ML Homelab Dashboard" in [heading.value for heading in app.title]
    assert [tab.label for tab in app.tabs] == ["S3", "Training", "Inference"]


def test_status_header_reports_services_online(app):
    app.run()
    markdown = [block.value for block in app.markdown]
    assert markdown.count("✅ Online") == 2
    assert "❌ Offline" not in markdown


def test_status_header_reports_services_offline(monkeypatch, s3_client):
    monkeypatch.setattr(storage, "create_client", lambda: s3_client)
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: False)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    assert [b.value for b in at.markdown].count("❌ Offline") == 2


def test_bucket_contents_are_listed(app):
    app.run()
    assert app.selectbox[0].options == ["models", "empty"]
    assert any("Contents of `models` (3 objects)" in b.value for b in app.markdown)
    assert any("`notes.txt`" in b.value for b in app.markdown)


def test_text_objects_are_previewed_inline(app):
    app.run()
    assert any("hello world" in block.value for block in app.code)


def test_object_sizes_are_rendered(app):
    app.run()
    rendered = [block.value for block in app.markdown]
    assert "4.0 KB" in rendered  # weights.bin


def test_switching_buckets_shows_the_empty_bucket(app):
    app.run()
    app.selectbox[0].set_value("empty").run()
    assert any("Contents of `empty` (0 objects)" in b.value for b in app.markdown)


def test_delete_button_removes_the_object(app, s3_client):
    app.run()
    delete_button = next(b for b in app.button if b.proto.id.endswith("delete-cat.png"))
    delete_button.click().run()

    assert ("models", "cat.png") in s3_client.deleted


def test_download_button_renders_a_presigned_link(app):
    app.run()
    download = next(b for b in app.button if b.proto.id.endswith("download-notes.txt"))
    download.click().run()

    assert any("Click to Download" in b.value for b in app.markdown)


def test_missing_buckets_shows_guidance(monkeypatch):
    monkeypatch.setattr(storage, "create_client", lambda: FakeS3Client({}))
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: False)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    assert any("No buckets found" in block.value for block in at.info)


def test_s3_connection_failure_is_surfaced(monkeypatch):
    class BrokenClient:
        def list_buckets(self):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(storage, "create_client", lambda: BrokenClient())
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: False)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    assert any("Failed to connect to S3 (Floci)" in e.value for e in at.error)


def test_object_listing_failure_is_surfaced(monkeypatch, s3_client):
    def broken_paginator(operation):
        raise RuntimeError("bucket unavailable")

    s3_client.get_paginator = broken_paginator
    monkeypatch.setattr(storage, "create_client", lambda: s3_client)
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: True)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    assert any("Error listing objects" in e.value for e in at.error)
    # The upload form still renders so the bucket is not a dead end.
    assert any("Upload a file" in b.value for b in at.markdown)


def test_a_broken_preview_does_not_look_like_a_listing_failure(monkeypatch, s3_client):
    """A corrupt image must blame the object, not the bucket listing."""
    s3_client.buckets["models"]["cat.png"] = b"not really a png"
    monkeypatch.setattr(storage, "create_client", lambda: s3_client)
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: True)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    assert any("Could not preview `cat.png`" in w.value for w in at.warning)
    assert not any("Error listing objects" in e.value for e in at.error)


def test_a_broken_preview_does_not_hide_the_remaining_objects(monkeypatch, s3_client):
    s3_client.buckets["models"]["cat.png"] = b"not really a png"
    monkeypatch.setattr(storage, "create_client", lambda: s3_client)
    monkeypatch.setattr(health, "is_port_open", lambda *a, **k: True)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()

    # weights.bin is listed after the broken cat.png.
    assert any("`weights.bin`" in b.value for b in at.markdown)


def test_upload_name_field_is_available_before_a_file_is_attached(app):
    """The field must exist on first render or it can never name the upload."""
    app.run()
    labels = [box.label for box in app.text_input]
    assert "Object name in S3" in labels


def test_job_expanders_are_rendered(app):
    app.run()
    labels = [expander.label for expander in app.expander]
    assert "Training Job - MNIST Tune" in labels
    assert "Inference Job - Resnet Inference" in labels


def test_running_a_job_streams_status_and_success(app, monkeypatch):
    client = FakeJobClient([JobStatus.RUNNING, JobStatus.SUCCEEDED])
    monkeypatch.setattr(job_runner, "create_client", lambda *a, **k: client)
    monkeypatch.setattr(job_runner.time, "sleep", lambda seconds: None)

    app.run()
    app.button(key="MNIST Tune").click().run()

    assert client.submitted[0]["entrypoint"] == "python mnist_training/train_mnist.py"
    successes = [block.value for block in app.success]
    assert any("submitted: `job_abc123`" in msg for msg in successes)
    assert any("finished successfully" in msg for msg in successes)


def test_failed_job_reports_an_error(app, monkeypatch):
    client = FakeJobClient([JobStatus.FAILED])
    monkeypatch.setattr(job_runner, "create_client", lambda *a, **k: client)
    monkeypatch.setattr(job_runner.time, "sleep", lambda seconds: None)

    app.run()
    app.button(key="Resnet Inference").click().run()

    assert any("ended with status" in block.value for block in app.error)


def test_app_script_imports_its_own_package_without_help(tmp_path):
    """`streamlit run` must be able to execute app.py directly.

    Streamlit puts the *script's* directory on ``sys.path``, not the repository
    root, so the absolute ``streamlit_app`` imports in ``app.py`` only resolve
    because the script bootstraps the root itself. Every other test here goes
    through ``AppTest`` under pytest, which supplies the root via
    ``pythonpath = ["."]`` and therefore cannot catch a regression.

    Run from an unrelated working directory with ``PYTHONPATH`` cleared so the
    only thing that can make the import work is the bootstrap.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, APP_PATH],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=APP_TIMEOUT * 2,
    )

    assert "No module named 'streamlit_app'" not in result.stderr
    assert result.returncode == 0, result.stderr[-2000:]
