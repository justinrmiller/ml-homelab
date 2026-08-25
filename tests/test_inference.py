"""Tests for the ResNet batch inference job."""

import numpy as np
import pytest
import torch
from PIL import Image

from streamlit_app.jobs.resnet_inference import inference
from tests.conftest import png_bytes


@pytest.fixture
def stub_model(monkeypatch):
    """A ResNetModel whose weights are a cheap deterministic stand-in."""

    class TinyNet(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros(batch, 1000)
            logits[:, 7] = 10.0  # class 7 always wins
            return logits

    monkeypatch.setitem(inference.MODEL_BUILDERS, "resnet50", TinyNet)
    return inference.ResNetModel(model_name="resnet50")


def test_unsupported_model_name_raises():
    with pytest.raises(ValueError, match="Unsupported model: resnet9000"):
        inference.ResNetModel(model_name="resnet9000")


def test_model_registry_covers_documented_variants():
    assert set(inference.MODEL_BUILDERS) == {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
    }


def test_transform_produces_expected_tensor_shape(stub_model):
    tensor = stub_model.transform(Image.new("RGB", (300, 400)))
    assert tensor.shape == (3, inference.IMAGE_SIZE, inference.IMAGE_SIZE)


def test_call_returns_predictions_for_each_image(stub_model):
    batch = {"bytes": [png_bytes(), png_bytes((0, 255, 0))], "path": ["a.png", "b.png"]}

    result = stub_model(batch)

    assert result["predicted_class"].tolist() == [7, 7]
    assert result["confidence_score"].shape == (2,)
    assert np.all(result["confidence_score"] > 0.9)
    assert result["top_5_classes"].shape == (2, 5)
    assert result["top_5_scores"].shape == (2, 5)
    assert result["file_path"] == ["a.png", "b.png"]


def test_call_substitutes_blank_tensor_for_undecodable_image(stub_model, capsys):
    batch = {"bytes": [b"not an image"], "path": ["broken.png"]}

    result = stub_model(batch)

    assert result["predicted_class"].shape == (1,)
    assert "Error processing image 0" in capsys.readouterr().out


def test_call_without_paths_pads_file_path(stub_model):
    result = stub_model({"bytes": [png_bytes()]})
    assert result["file_path"] == [None]


def test_filter_image_files_keeps_only_images():
    batch = {
        "path": ["a.jpg", "b.txt", "c.PNG", "d.parquet", "e.webp"],
        "bytes": [b"1", b"2", b"3", b"4", b"5"],
    }

    filtered = inference.filter_image_files(batch)

    assert filtered["path"] == ["a.jpg", "c.PNG", "e.webp"]
    assert filtered["bytes"] == [b"1", b"3", b"5"]


def test_filter_image_files_returns_empty_batch_when_no_images():
    batch = {"path": ["a.txt", "b.parquet"], "bytes": [b"1", b"2"]}
    assert inference.filter_image_files(batch) == {"path": [], "bytes": []}


def test_filter_image_files_handles_empty_input():
    assert inference.filter_image_files({"path": [], "bytes": []}) == {
        "path": [],
        "bytes": [],
    }


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("out.parquet", "parquet"),
        ("out.json", "json"),
        ("out.csv", "csv"),
        ("./predictions/", "parquet"),
        ("./predictions", "parquet"),
        ("out.unknown", "parquet"),
    ],
)
def test_output_format(path, expected):
    assert inference.output_format(path) == expected


class RecordingDataset:
    """Records which Ray Data writer was invoked."""

    def __init__(self):
        self.calls = []

    def write_parquet(self, path):
        self.calls.append(("parquet", path))

    def write_json(self, path):
        self.calls.append(("json", path))

    def write_csv(self, path):
        self.calls.append(("csv", path))


@pytest.mark.parametrize(
    ("path", "expected"),
    [("out.parquet", "parquet"), ("out.json", "json"), ("out.csv", "csv")],
)
def test_save_predictions_locally_dispatches_writer(path, expected):
    dataset = RecordingDataset()
    inference.save_predictions_locally(dataset, path)  # ty: ignore[invalid-argument-type]
    assert dataset.calls == [(expected, path)]


def test_load_images_from_s3_omits_filesystem_without_credentials(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        inference.ray.data,
        "read_binary_files",
        lambda uri, filesystem, include_paths: captured.update(
            uri=uri, filesystem=filesystem, include_paths=include_paths
        ),
    )

    inference.load_images_from_s3("s3://bucket/images/")

    assert captured["filesystem"] is None
    assert captured["include_paths"] is True


def test_load_images_from_s3_passes_credentials(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        inference.ray.data,
        "read_binary_files",
        lambda uri, filesystem, include_paths: captured.update(filesystem=filesystem),
    )

    inference.load_images_from_s3(
        "s3://bucket/images/",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        aws_session_token="token",
    )

    assert captured["filesystem"] == {
        "access_key_id": "key",
        "secret_access_key": "secret",
        "session_token": "token",
    }


def test_load_images_from_s3_omits_absent_session_token(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        inference.ray.data,
        "read_binary_files",
        lambda uri, filesystem, include_paths: captured.update(filesystem=filesystem),
    )

    inference.load_images_from_s3(
        "s3://bucket/images/", aws_access_key_id="key", aws_secret_access_key="secret"
    )

    assert "session_token" not in captured["filesystem"]


def test_show_sample_predictions_prints_each_sample(capsys):
    class SampleDataset:
        def take(self, n):
            return [
                {
                    "file_path": "/data/cat.jpg",
                    "predicted_class": 281,
                    "confidence_score": 0.9312,
                    "top_5_classes": np.array([281, 282, 283, 284, 285]),
                    "top_5_scores": np.array([0.93, 0.02, 0.01, 0.01, 0.005]),
                }
            ]

    inference.show_sample_predictions(SampleDataset(), num_samples=1)  # ty: ignore[invalid-argument-type]

    out = capsys.readouterr().out
    assert "File: cat.jpg" in out
    assert "Predicted class: 281" in out
    assert "Confidence: 0.9312" in out


class FakeDataset:
    """Records the lazy operations a Ray Dataset would have been asked to do."""

    def __init__(self, calls):
        self.calls = calls

    def count(self):
        self.calls.setdefault("count", 0)
        self.calls["count"] += 1
        return 3

    def map_batches(self, fn, **kwargs):
        self.calls.setdefault("map_batches", []).append((fn, kwargs))
        return self

    def materialize(self):
        self.calls.setdefault("materialize", 0)
        self.calls["materialize"] += 1
        return self


def test_run_resnet_batch_prediction_wires_the_pipeline(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(
        inference, "load_images_from_s3", lambda **kwargs: FakeDataset(calls)
    )

    result = inference.run_resnet_batch_prediction(
        "s3://bucket/images/", model_name="resnet18", batch_size=8, num_gpus=0.0
    )

    assert isinstance(result, FakeDataset)
    filter_call, model_call = calls["map_batches"]
    assert filter_call[0] is inference.filter_image_files
    assert model_call[0] is inference.ResNetModel
    assert model_call[1]["fn_constructor_kwargs"] == {"model_name": "resnet18"}
    assert model_call[1]["batch_size"] == 8
    assert "Running batch inference with resnet18" in capsys.readouterr().out


def test_run_resnet_batch_prediction_materializes_once(monkeypatch):
    """Without materialize(), count/take/write each re-run inference."""
    calls = {}
    monkeypatch.setattr(
        inference, "load_images_from_s3", lambda **kwargs: FakeDataset(calls)
    )

    inference.run_resnet_batch_prediction("s3://bucket/images/")

    assert calls["materialize"] == 1


def test_run_resnet_batch_prediction_does_not_count_eagerly(monkeypatch):
    """Each count() on a lazy dataset would force another full read of S3."""
    calls = {}
    monkeypatch.setattr(
        inference, "load_images_from_s3", lambda **kwargs: FakeDataset(calls)
    )

    inference.run_resnet_batch_prediction("s3://bucket/images/")

    assert "count" not in calls


def test_parse_args_defaults():
    args = inference.parse_args([])
    assert args.model_name == "resnet50"
    assert args.batch_size == 64
    assert args.num_gpus == 0.0
    assert args.smoke_test is False


def test_parse_args_overrides():
    args = inference.parse_args(
        ["--model-name", "resnet18", "--batch-size", "8", "--smoke-test"]
    )
    assert args.model_name == "resnet18"
    assert args.batch_size == 8
    assert args.smoke_test is True


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit):
        inference.parse_args(["--model-name", "vgg16"])


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Replace Ray and the prediction pipeline, recording the calls made."""
    calls = {}

    class FakePredictions:
        def count(self):
            return 42

    monkeypatch.setattr(inference.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(inference.ray, "init", lambda: calls.setdefault("init", True))
    monkeypatch.setattr(
        inference.ray, "shutdown", lambda: calls.setdefault("shutdown", True)
    )

    def fake_predict(**kwargs):
        calls["predict"] = kwargs
        return FakePredictions()

    monkeypatch.setattr(inference, "run_resnet_batch_prediction", fake_predict)
    monkeypatch.setattr(
        inference,
        "show_sample_predictions",
        lambda predictions, num: calls.setdefault("shown", num),
    )
    monkeypatch.setattr(
        inference,
        "save_predictions_locally",
        lambda predictions, output_path: calls.setdefault("saved", output_path),
    )
    return calls


def test_main_runs_the_pipeline_and_shuts_ray_down(stubbed_pipeline, capsys):
    inference.main(["--s3-uri", "s3://bucket/imgs/", "--output-path", "out.csv"])

    assert stubbed_pipeline["init"] is True
    assert stubbed_pipeline["predict"]["s3_uri"] == "s3://bucket/imgs/"
    assert stubbed_pipeline["saved"] == "out.csv"
    assert stubbed_pipeline["shutdown"] is True
    assert "Total predictions: 42" in capsys.readouterr().out


def test_main_caps_batch_size_for_smoke_test(stubbed_pipeline):
    inference.main(["--batch-size", "128", "--smoke-test"])
    assert stubbed_pipeline["predict"]["batch_size"] == 4


def test_main_keeps_batch_size_without_smoke_test(stubbed_pipeline):
    inference.main(["--batch-size", "16"])
    assert stubbed_pipeline["predict"]["batch_size"] == 16


def test_main_reraises_pipeline_failures(monkeypatch, capsys):
    monkeypatch.setattr(inference.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.ray, "shutdown", lambda: None)

    def boom(**kwargs):
        raise RuntimeError("cluster unreachable")

    monkeypatch.setattr(inference, "run_resnet_batch_prediction", boom)

    with pytest.raises(RuntimeError, match="cluster unreachable"):
        inference.main([])

    assert "Error during batch prediction" in capsys.readouterr().out


def test_main_leaves_a_ray_session_it_did_not_start_running(monkeypatch):
    """As a submitted Ray job the driver connection is not ours to close."""
    shutdowns = []
    monkeypatch.setattr(inference.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.ray, "init", lambda: pytest.fail("must not init"))
    monkeypatch.setattr(inference.ray, "shutdown", lambda: shutdowns.append(True))
    monkeypatch.setattr(
        inference, "run_resnet_batch_prediction", lambda **kwargs: FakeDataset({})
    )
    monkeypatch.setattr(inference, "show_sample_predictions", lambda *a, **k: None)
    monkeypatch.setattr(inference, "save_predictions_locally", lambda **kwargs: None)

    inference.main([])

    assert shutdowns == []


def test_main_shuts_down_the_ray_session_it_started(stubbed_pipeline):
    inference.main([])
    assert stubbed_pipeline["shutdown"] is True


def test_show_sample_predictions_handles_missing_file_path(capsys):
    class SampleDataset:
        def take(self, n):
            return [
                {
                    "file_path": None,
                    "predicted_class": 1,
                    "confidence_score": 0.5,
                    "top_5_classes": np.arange(5),
                    "top_5_scores": np.linspace(0.5, 0.1, 5),
                }
            ]

    inference.show_sample_predictions(SampleDataset(), num_samples=1)  # ty: ignore[invalid-argument-type]
    assert "File:" not in capsys.readouterr().out
