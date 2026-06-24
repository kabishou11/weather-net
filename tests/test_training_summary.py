from src.weather_net.metrics import ClassificationReport
from src.weather_net.training import build_fold_summary


def test_build_fold_summary_includes_report_details() -> None:
    report = ClassificationReport(
        macro_f1=0.75,
        accuracy=0.8,
        per_class_f1={"rain": 0.7, "sunny": 0.8},
        confusion_matrix=[[3, 1], [0, 4]],
    )

    summary = build_fold_summary(
        fold=1,
        best_epoch=3,
        best_val_loss=0.42,
        checkpoint="outputs/model_fold1.pt",
        report=report,
    )

    assert summary == {
        "fold": 1,
        "best_epoch": 3,
        "best_macro_f1": 0.75,
        "best_accuracy": 0.8,
        "best_val_loss": 0.42,
        "checkpoint": "outputs/model_fold1.pt",
        "per_class_f1": {"rain": 0.7, "sunny": 0.8},
        "confusion_matrix": [[3, 1], [0, 4]],
    }


def test_build_fold_summary_can_include_oof_artifacts() -> None:
    from src.weather_net.oof import OofArtifactPaths

    report = ClassificationReport(
        macro_f1=0.75,
        accuracy=0.8,
        per_class_f1={"rain": 0.7, "sunny": 0.8},
        confusion_matrix=[[3, 1], [0, 4]],
    )

    summary = build_fold_summary(
        fold=1,
        best_epoch=3,
        best_val_loss=0.42,
        checkpoint="outputs/model_fold1.pt",
        report=report,
        oof_artifacts=OofArtifactPaths(
            csv_path="outputs/oof/oof_predictions.csv",
            npz_path="outputs/oof/oof_probabilities.npz",
            metrics_path="outputs/oof/oof_metrics.json",
        ),
    )

    assert summary["oof_predictions_csv"] == "outputs/oof/oof_predictions.csv"
    assert summary["oof_probabilities_npz"] == "outputs/oof/oof_probabilities.npz"
    assert summary["oof_metrics_json"] == "outputs/oof/oof_metrics.json"


def test_collect_oof_predictions_returns_logits_and_ids(tmp_path) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.weather_net.data import ManifestRow
    from src.weather_net.training import collect_oof_predictions

    class FixedModel(nn.Module):
        def forward(self, images):
            return torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=images.dtype)

    rows = [
        ManifestRow(
            path=tmp_path / "rain.jpg",
            label=0,
            label_name="rain",
            image_id="rain.jpg",
            source="labeled",
        ),
        ManifestRow(
            path=tmp_path / "sunny.jpg",
            label=1,
            label_name="sunny",
            image_id="sunny.jpg",
            source="labeled",
        ),
    ]
    dataset = TensorDataset(
        torch.zeros(2, 3, 4, 4),
        torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]),
    )
    loader = DataLoader(dataset, batch_size=2)

    records = collect_oof_predictions(
        model=FixedModel(),
        rows=rows,
        class_names=["rain", "sunny"],
        fold=0,
        checkpoint="fold0.pt",
        model_name="fixed",
        device="cpu",
        loader=loader,
    )

    assert [record.image_id for record in records] == ["rain.jpg", "sunny.jpg"]
    assert [record.true_label for record in records] == ["rain", "sunny"]
    assert [list(record.logits) for record in records] == [[2.0, 0.0], [0.0, 2.0]]


def test_train_config_passes_optimizer_group_options(monkeypatch, tmp_path) -> None:
    import torch
    from torch import nn

    from src.weather_net.config import AppConfig
    from src.weather_net.data import ManifestRow
    from src.weather_net.metrics import ClassificationReport
    from src.weather_net import training

    seen: dict[str, object] = {}

    def fake_manifest(_config):
        rows = [
            ManifestRow(path=tmp_path / "rain.jpg", label=0, label_name="rain", image_id="rain.jpg"),
            ManifestRow(path=tmp_path / "sunny.jpg", label=1, label_name="sunny", image_id="sunny.jpg"),
        ]
        return rows, {"rain": 0, "sunny": 1}

    def fake_loaders(train_rows, val_rows, **_kwargs):
        return object(), object()

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(1, 2)

    def fake_create_classifier(*_args, **_kwargs):
        return TinyModel()

    def fake_build_optimizer(model, lr, weight_decay, no_weight_decay=False, layer_decay=1.0):
        seen.update(
            {
                "model": model,
                "lr": lr,
                "weight_decay": weight_decay,
                "no_weight_decay": no_weight_decay,
                "layer_decay": layer_decay,
            }
        )
        return torch.optim.SGD(model.parameters(), lr=lr)

    monkeypatch.setattr(training, "load_training_manifest", fake_manifest)
    monkeypatch.setattr(training, "make_loaders", fake_loaders)
    monkeypatch.setattr(training, "create_classifier", fake_create_classifier)
    monkeypatch.setattr(training, "build_optimizer", fake_build_optimizer)
    monkeypatch.setattr(training, "train_one_epoch", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(
        training,
        "evaluate",
        lambda *_args, **_kwargs: (
            ClassificationReport(
                macro_f1=1.0,
                accuracy=1.0,
                per_class_f1={"rain": 1.0, "sunny": 1.0},
                confusion_matrix=[[1, 0], [0, 1]],
            ),
            0.1,
        ),
    )
    monkeypatch.setattr(training, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training, "collect_oof_predictions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(training, "write_oof_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training.torch, "load", lambda *_args, **_kwargs: {"model_state": TinyModel().state_dict()})

    class NoopScheduler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def step(self) -> None:
            pass

    monkeypatch.setattr(training.torch.optim.lr_scheduler, "CosineAnnealingLR", NoopScheduler)

    config = AppConfig()
    config.data.folds = 1
    config.train.epochs = 1
    config.train.output_dir = tmp_path / "outputs"
    config.train.no_weight_decay = True
    config.train.layer_decay = 0.85

    training.train_config(config, device_request="cpu")

    assert seen["no_weight_decay"] is True
    assert seen["layer_decay"] == 0.85
