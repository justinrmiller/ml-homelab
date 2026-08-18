"""Shared fixtures: in-memory doubles for the S3 and Ray Jobs clients."""

import io
from collections.abc import Iterator
from typing import Any

import pytest
from PIL import Image
from ray.job_submission import JobStatus


def png_bytes(color: tuple[int, int, int] = (255, 0, 0), size: int = 8) -> bytes:
    """Encode a small solid-colour PNG, so image previews get real bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakePaginator:
    """Pages a bucket listing the way botocore's paginator does."""

    def __init__(self, client: "FakeS3Client", page_size: int):
        """Page over ``client``'s objects ``page_size`` keys at a time."""
        self.client = client
        self.page_size = page_size

    def paginate(self, Bucket: str) -> Iterator[dict[str, Any]]:
        """Yield one response page at a time, omitting ``Contents`` when empty."""
        contents = [
            {"Key": key, "Size": len(body)}
            for key, body in self.client.buckets[Bucket].items()
        ]
        if not contents:
            yield {}
            return
        for start in range(0, len(contents), self.page_size):
            yield {"Contents": contents[start : start + self.page_size]}


class FakeS3Client:
    """Minimal in-memory stand-in for the boto3 S3 client.

    ``page_size`` defaults to 2 so that any bucket with more than two objects
    exercises the pagination path rather than hiding it.
    """

    def __init__(
        self, buckets: dict[str, dict[str, bytes]] | None = None, page_size: int = 2
    ):
        """Seed the fake with ``{bucket: {key: body}}``."""
        self.buckets = buckets if buckets is not None else {}
        self.page_size = page_size
        self.uploads: list[tuple[str, str, bytes]] = []
        self.deleted: list[tuple[str, str]] = []

    def list_buckets(self) -> dict[str, Any]:
        """Return bucket metadata in the shape boto3 uses."""
        return {"Buckets": [{"Name": name} for name in self.buckets]}

    def get_paginator(self, operation: str) -> FakePaginator:
        """Return a paginator for ``list_objects_v2``."""
        assert operation == "list_objects_v2", operation
        return FakePaginator(self, self.page_size)

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        """Return an object body wrapped in a file-like object."""
        return {"Body": io.BytesIO(self.buckets[Bucket][Key])}

    def generate_presigned_url(
        self,
        operation: str,
        Params: dict[str, str],
        ExpiresIn: int,
    ) -> str:
        """Return a deterministic fake presigned URL."""
        return (
            f"https://minio.test/{operation}/{Params['Bucket']}/{Params['Key']}"
            f"?expires={ExpiresIn}"
        )

    def delete_object(self, Bucket: str, Key: str) -> None:
        """Delete an object and record the call."""
        del self.buckets[Bucket][Key]
        self.deleted.append((Bucket, Key))

    def upload_fileobj(self, fileobj: Any, bucket: str, key: str) -> None:
        """Store an uploaded file body and record the call."""
        body = fileobj.read()
        self.buckets.setdefault(bucket, {})[key] = body
        self.uploads.append((bucket, key, body))


class FakeJobClient:
    """Jobs API stand-in that walks through a scripted status sequence."""

    def __init__(self, statuses: list[JobStatus], logs: str = "log output"):
        """Configure the statuses returned by successive polls."""
        self.statuses = list(statuses)
        self.logs = logs
        self.submitted: list[dict[str, Any]] = []
        self.status_calls = 0
        self.log_calls = 0

    def submit_job(self, entrypoint: str, runtime_env: dict[str, Any]) -> str:
        """Record a submission and return a fixed job id."""
        self.submitted.append({"entrypoint": entrypoint, "runtime_env": runtime_env})
        return "job_abc123"

    def get_job_status(self, job_id: str) -> JobStatus:
        """Return the next scripted status, repeating the last one."""
        self.status_calls += 1
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def get_job_logs(self, job_id: str) -> str:
        """Return the canned log blob."""
        self.log_calls += 1
        return self.logs


@pytest.fixture
def s3_client() -> FakeS3Client:
    """An S3 double seeded with one bucket of mixed object types."""
    return FakeS3Client(
        {
            "models": {
                "notes.txt": b"hello world",
                "cat.png": png_bytes(),
                "weights.bin": b"0" * 4096,
            },
            "empty": {},
        }
    )


@pytest.fixture
def sleepless() -> tuple[list[float], Any]:
    """A sleep function that records its calls instead of waiting."""
    calls: list[float] = []
    return calls, calls.append
