"""Floci/S3 helpers used by the dashboard's storage browser."""

import os
from typing import Any, BinaryIO, Literal

import boto3

from streamlit_app import health

# Derived from health.FLOCI_PORT rather than hardcoded, so that setting
# FLOCI_PORT moves the compose port mapping, the status check, and this client
# together instead of only the first.
DEFAULT_ENDPOINT_URL = f"http://localhost:{health.FLOCI_PORT}"
DEFAULT_ACCESS_KEY = "test"
DEFAULT_SECRET_KEY = "test"

TEXT_PREVIEW_EXTENSIONS = (".txt", ".csv", ".json")
IMAGE_PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg")
TEXT_PREVIEW_LIMIT = 500
PRESIGNED_URL_TTL = 3600

PreviewKind = Literal["text", "image"]

KIB = 1024
MIB = 1024**2


def create_client() -> Any:
    """Build an S3 client pointed at the configured Floci endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3", DEFAULT_ENDPOINT_URL),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", DEFAULT_ACCESS_KEY),
        aws_secret_access_key=os.environ.get(
            "AWS_SECRET_ACCESS_KEY", DEFAULT_SECRET_KEY
        ),
    )


def list_bucket_names(client: Any) -> list[str]:
    """Return the names of every bucket visible to ``client``."""
    return [bucket["Name"] for bucket in client.list_buckets()["Buckets"]]


def list_objects(client: Any, bucket: str) -> list[dict[str, Any]]:
    """Return every object stored in ``bucket``.

    ``list_objects_v2`` returns at most 1000 keys per call, so this pages
    through the whole bucket rather than silently truncating the listing.

    Args:
        client: An S3 client.
        bucket: The bucket to list.

    Returns:
        Every object's metadata, or an empty list for an empty bucket.
    """
    objects: list[dict[str, Any]] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        objects.extend(page.get("Contents", []))
    return objects


def format_size(size_bytes: int) -> str:
    """Format a byte count as KB below 1 MiB and MB above it."""
    if size_bytes < MIB:
        return f"{size_bytes / KIB:.1f} KB"
    return f"{size_bytes / MIB:.1f} MB"


def preview_kind(key: str) -> PreviewKind | None:
    """Classify how an object should be previewed based on its extension.

    Args:
        key: The object key.

    Returns:
        ``"text"``, ``"image"``, or None when the type has no inline preview.
    """
    lowered = key.lower()
    if lowered.endswith(TEXT_PREVIEW_EXTENSIONS):
        return "text"
    if lowered.endswith(IMAGE_PREVIEW_EXTENSIONS):
        return "image"
    return None


def read_object_bytes(client: Any, bucket: str, key: str) -> bytes:
    """Read an object's full body as bytes."""
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_text_preview(
    client: Any, bucket: str, key: str, limit: int = TEXT_PREVIEW_LIMIT
) -> str:
    """Read the first ``limit`` characters of a text object, ignoring bad bytes."""
    body = read_object_bytes(client, bucket, key)
    return body.decode("utf-8", errors="replace")[:limit]


def presigned_download_url(
    client: Any, bucket: str, key: str, expires_in: int = PRESIGNED_URL_TTL
) -> str:
    """Generate a time limited download URL for an object."""
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def delete_object(client: Any, bucket: str, key: str) -> None:
    """Delete a single object from a bucket."""
    client.delete_object(Bucket=bucket, Key=key)


def upload_fileobj(client: Any, fileobj: BinaryIO, bucket: str, key: str) -> None:
    """Upload an open file object to ``bucket`` under ``key``."""
    client.upload_fileobj(fileobj, bucket, key)


def resolve_object_key(dest_name: str, filename: str) -> str:
    """Pick the destination key for an upload.

    Args:
        dest_name: The name the user typed, which may be blank.
        filename: The name of the file they selected.

    Returns:
        The trimmed name the user typed, falling back to the file's own name.
    """
    return dest_name.strip() or filename


def try_upload(
    client: Any, fileobj: BinaryIO, bucket: str, key: str
) -> tuple[bool, str]:
    """Upload a file, converting any failure into a message for the UI.

    Args:
        client: An S3 client.
        fileobj: The open file to upload.
        bucket: Destination bucket.
        key: Destination object key.

    Returns:
        ``(succeeded, message)`` — the message is always safe to display.
    """
    try:
        upload_fileobj(client, fileobj, bucket, key)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        return False, f"Upload failed: {exc}"
    return True, f"Uploaded `{key}` to `{bucket}`"
