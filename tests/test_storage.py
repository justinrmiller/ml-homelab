"""Tests for the MinIO/S3 helper layer."""

import io

import pytest

from streamlit_app import storage
from tests.conftest import FakeS3Client


def test_create_client_uses_environment(monkeypatch):
    captured = {}

    def fake_boto_client(service, **kwargs):
        captured["service"] = service
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(storage.boto3, "client", fake_boto_client)
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://minio:9000")
    monkeypatch.setenv("MINIO_ROOT_USER", "user")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "secret")

    storage.create_client()

    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "http://minio:9000"
    assert captured["aws_access_key_id"] == "user"
    assert captured["aws_secret_access_key"] == "secret"


def test_create_client_falls_back_to_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        storage.boto3, "client", lambda service, **kwargs: captured.update(kwargs)
    )
    for var in ("AWS_ENDPOINT_URL_S3", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    storage.create_client()

    assert captured["endpoint_url"] == storage.DEFAULT_ENDPOINT_URL
    assert captured["aws_access_key_id"] == storage.DEFAULT_ACCESS_KEY
    assert captured["aws_secret_access_key"] == storage.DEFAULT_SECRET_KEY


def test_list_bucket_names(s3_client):
    assert storage.list_bucket_names(s3_client) == ["models", "empty"]


def test_list_objects_returns_metadata(s3_client):
    objects = storage.list_objects(s3_client, "models")
    assert [obj["Key"] for obj in objects] == ["notes.txt", "cat.png", "weights.bin"]
    assert objects[0]["Size"] == len(b"hello world")


def test_list_objects_handles_empty_bucket(s3_client):
    assert storage.list_objects(s3_client, "empty") == []


def test_list_objects_pages_through_the_whole_bucket():
    """``list_objects_v2`` caps at 1000 keys, so the listing must paginate."""
    client = FakeS3Client(
        {"big": {f"obj-{i}": b"x" for i in range(2500)}}, page_size=1000
    )

    objects = storage.list_objects(client, "big")

    assert len(objects) == 2500
    assert objects[-1]["Key"] == "obj-2499"


def test_list_objects_spans_a_partial_final_page():
    client = FakeS3Client({"big": {f"obj-{i}": b"x" for i in range(5)}}, page_size=2)
    assert len(storage.list_objects(client, "big")) == 5


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0.0 KB"),
        (512, "0.5 KB"),
        (storage.KIB, "1.0 KB"),
        (storage.MIB - 1, "1024.0 KB"),
        (storage.MIB, "1.0 MB"),
        (3 * storage.MIB, "3.0 MB"),
    ],
)
def test_format_size(size, expected):
    assert storage.format_size(size) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("a.txt", "text"),
        ("a.csv", "text"),
        ("a.json", "text"),
        ("A.JSON", "text"),
        ("a.png", "image"),
        ("a.jpg", "image"),
        ("photo.JPEG", "image"),
        ("model.bin", None),
        ("no-extension", None),
        ("txt", None),
    ],
)
def test_preview_kind(key, expected):
    assert storage.preview_kind(key) == expected


def test_read_object_bytes(s3_client):
    assert storage.read_object_bytes(s3_client, "models", "notes.txt") == b"hello world"


def test_read_text_preview_truncates(s3_client):
    s3_client.buckets["models"]["big.txt"] = b"x" * 1000
    preview = storage.read_text_preview(s3_client, "models", "big.txt", limit=10)
    assert preview == "x" * 10


def test_read_text_preview_survives_invalid_utf8(s3_client):
    s3_client.buckets["models"]["bad.txt"] = b"ok \xff\xfe"
    assert storage.read_text_preview(s3_client, "models", "bad.txt").startswith("ok ")


def test_presigned_download_url(s3_client):
    url = storage.presigned_download_url(s3_client, "models", "notes.txt")
    assert url == (
        "https://minio.test/get_object/models/notes.txt"
        f"?expires={storage.PRESIGNED_URL_TTL}"
    )


def test_delete_object(s3_client):
    storage.delete_object(s3_client, "models", "notes.txt")
    assert "notes.txt" not in s3_client.buckets["models"]
    assert s3_client.deleted == [("models", "notes.txt")]


def test_upload_fileobj(s3_client):
    storage.upload_fileobj(s3_client, io.BytesIO(b"payload"), "models", "new.txt")
    assert s3_client.buckets["models"]["new.txt"] == b"payload"
    assert s3_client.uploads == [("models", "new.txt", b"payload")]


def test_try_upload_reports_success(s3_client):
    ok, message = storage.try_upload(
        s3_client, io.BytesIO(b"payload"), "models", "new.txt"
    )
    assert ok is True
    assert message == "Uploaded `new.txt` to `models`"


def test_try_upload_converts_failures_into_messages(s3_client):
    def boom(fileobj, bucket, key):
        raise RuntimeError("bucket is read-only")

    s3_client.upload_fileobj = boom

    ok, message = storage.try_upload(
        s3_client, io.BytesIO(b"payload"), "models", "new.txt"
    )

    assert ok is False
    assert message == "Upload failed: bucket is read-only"


@pytest.mark.parametrize(
    ("typed", "filename", "expected"),
    [
        ("", "photo.png", "photo.png"),
        ("   ", "photo.png", "photo.png"),
        ("renamed.png", "photo.png", "renamed.png"),
        ("  renamed.png  ", "photo.png", "renamed.png"),
    ],
)
def test_resolve_object_key(typed, filename, expected):
    assert storage.resolve_object_key(typed, filename) == expected
