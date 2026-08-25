"""Tests for the MNIST tuning job."""

import pytest
import torch

from streamlit_app.jobs.mnist_training import train_mnist


@pytest.fixture
def tiny_loader():
    """A two-batch loader of 8x8-ish MNIST-shaped tensors."""
    images = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    return [(images, targets), (images, targets)]


def test_convnet_output_shape():
    model = train_mnist.ConvNet()
    output = model(torch.randn(5, 1, 28, 28))
    assert output.shape == (5, 10)


def test_convnet_returns_log_probabilities():
    model = train_mnist.ConvNet()
    output = model(torch.randn(3, 1, 28, 28))
    # log_softmax rows exponentiate to 1.
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(3), atol=1e-5)


def test_train_func_updates_parameters(tiny_loader):
    model = train_mnist.ConvNet()
    before = model.fc.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    train_mnist.train_func(model, optimizer, tiny_loader)

    assert not torch.equal(before, model.fc.weight.detach())


def test_train_func_stops_once_epoch_size_is_exceeded(monkeypatch):
    """Batches are consumed until ``batch_idx * batch_size`` passes EPOCH_SIZE."""
    monkeypatch.setattr(train_mnist, "EPOCH_SIZE", 4)
    images = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    loader = [(images, targets)] * 10

    model = train_mnist.ConvNet()
    optimizer = CountingOptimizer(model.parameters())

    train_mnist.train_func(model, optimizer, loader)

    # batch 0 -> 0, batch 1 -> 4 (not > 4), batch 2 -> 8 (> 4, stop).
    assert optimizer.steps == 2


class CountingOptimizer(torch.optim.SGD):
    """SGD that records how many optimisation steps were taken."""

    def __init__(self, params):
        super().__init__(params, lr=0.5)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_test_func_returns_accuracy_between_zero_and_one(tiny_loader):
    accuracy = train_mnist.test_func(train_mnist.ConvNet(), tiny_loader)
    assert 0.0 <= accuracy <= 1.0


def test_test_func_scores_a_perfect_model():
    targets = torch.tensor([3, 3])
    images = torch.randn(2, 1, 28, 28)

    class AlwaysThree(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 10)
            logits[:, 3] = 1.0
            return logits

    assert train_mnist.test_func(AlwaysThree(), [(images, targets)]) == 1.0


def test_test_func_leaves_model_in_eval_mode(tiny_loader):
    model = train_mnist.ConvNet()
    model.train()
    train_mnist.test_func(model, tiny_loader)
    assert model.training is False


@pytest.mark.parametrize(
    ("argv", "cuda", "smoke"),
    [
        ([], False, False),
        (["--cuda"], True, False),
        (["--smoke-test"], False, True),
        (["--cuda", "--smoke-test"], True, True),
    ],
)
def test_parse_args(argv, cuda, smoke):
    args = train_mnist.parse_args(argv)
    assert args.cuda is cuda
    assert args.smoke_test is smoke


def test_parse_args_ignores_unknown_flags():
    assert train_mnist.parse_args(["--unrecognized", "value"]).cuda is False


def test_build_tuner_scales_down_for_smoke_test():
    tuner = train_mnist.build_tuner(train_mnist.parse_args(["--smoke-test"]))
    assert tuner is not None


def test_main_reports_trial_errors(monkeypatch):
    class FailedResults:
        errors = ["trial blew up"]

        def get_best_result(self):
            return type("R", (), {"config": {"lr": 0.1}})()

    monkeypatch.setattr(
        train_mnist,
        "build_tuner",
        lambda args: type("T", (), {"fit": lambda self: FailedResults()})(),
    )

    with pytest.raises(RuntimeError, match="1 trial"):
        train_mnist.main([])


def test_main_succeeds_when_no_trials_error(monkeypatch, capsys):
    class Results:
        errors = []

        def get_best_result(self):
            return type("R", (), {"config": {"lr": 0.01, "momentum": 0.5}})()

    monkeypatch.setattr(
        train_mnist,
        "build_tuner",
        lambda args: type("T", (), {"fit": lambda self: Results()})(),
    )

    train_mnist.main(["--smoke-test"])

    assert "Best config is:" in capsys.readouterr().out


def test_train_mnist_reports_metrics(monkeypatch, tiny_loader):
    reported = []
    monkeypatch.setattr(
        train_mnist, "get_data_loaders", lambda *a, **k: (tiny_loader, tiny_loader)
    )

    def fake_report(metrics, checkpoint=None):
        reported.append(metrics)
        raise StopIteration  # break out of the infinite training loop

    monkeypatch.setattr(train_mnist.tune, "report", fake_report)

    with pytest.raises(StopIteration):
        train_mnist.train_mnist({"lr": 0.01, "momentum": 0.5})

    assert "mean_accuracy" in reported[0]


def test_train_mnist_writes_checkpoint_when_requested(monkeypatch, tiny_loader):
    saved = []
    monkeypatch.setattr(
        train_mnist, "get_data_loaders", lambda *a, **k: (tiny_loader, tiny_loader)
    )

    def fake_report(metrics, checkpoint=None):
        saved.append(checkpoint)
        raise StopIteration

    monkeypatch.setattr(train_mnist.tune, "report", fake_report)

    with pytest.raises(StopIteration):
        train_mnist.train_mnist(
            {"lr": 0.01, "momentum": 0.5, "should_checkpoint": True}
        )

    assert saved[0] is not None


def test_test_func_stops_once_test_size_is_exceeded(monkeypatch):
    """Evaluation stops early, so accuracy is computed over a bounded sample."""
    monkeypatch.setattr(train_mnist, "TEST_SIZE", 4)
    images = torch.randn(4, 1, 28, 28)
    correct_targets = torch.tensor([3, 3, 3, 3])
    wrong_targets = torch.tensor([0, 0, 0, 0])

    class AlwaysThree(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 10)
            logits[:, 3] = 1.0
            return logits

    loader = [
        (images, correct_targets),
        (images, correct_targets),
        (images, wrong_targets),  # never reached: batch 2 -> 8 > 4
    ]

    assert train_mnist.test_func(AlwaysThree(), loader) == 1.0


def test_get_data_loaders_builds_train_and_test_loaders(monkeypatch, tmp_path):
    built = []

    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            built.append({"root": root, "train": train, "download": download})
            self.transform = transform

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), 0

    monkeypatch.setattr(train_mnist.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(
        train_mnist.os.path, "expanduser", lambda p: str(tmp_path / "data.lock")
    )

    train_loader, test_loader = train_mnist.get_data_loaders(batch_size=16)

    assert [entry["train"] for entry in built] == [True, False]
    assert all(entry["download"] for entry in built)
    assert train_loader.batch_size == 16
    assert test_loader.batch_size == 16
