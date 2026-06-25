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


def test_write_training_history_artifacts_outputs_csv_json_and_png(tmp_path) -> None:
    import csv
    import json

    from src.weather_net.training import write_training_history_artifacts

    history = [
        {
            "fold": 0,
            "epoch": 1,
            "loss_name": "class_balanced_focal",
            "effective_loss_name": "ce",
            "train_loss": 0.8,
            "val_loss": 0.7,
            "macro_f1": 0.55,
            "accuracy": 0.6,
            "ema": True,
            "seconds": 12.0,
            "best_so_far": True,
        },
        {
            "fold": 0,
            "epoch": 2,
            "loss_name": "class_balanced_focal",
            "effective_loss_name": "class_balanced_focal",
            "train_loss": 0.5,
            "val_loss": 0.6,
            "macro_f1": 0.7,
            "accuracy": 0.72,
            "ema": True,
            "seconds": 11.0,
            "best_so_far": True,
        },
    ]

    artifacts = write_training_history_artifacts(tmp_path, history)

    assert artifacts["csv"] == str(tmp_path / "training_history.csv")
    assert artifacts["json"] == str(tmp_path / "training_history.json")
    assert artifacts["curve_png"] == str(tmp_path / "training_curves.png")
    with (tmp_path / "training_history.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fold"] == "0"
    assert rows[1]["macro_f1"] == "0.700000"
    payload = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert payload[1]["epoch"] == 2
    assert (tmp_path / "training_curves.png").read_bytes().startswith(b"\x89PNG")


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


def test_train_config_writes_history_artifacts_and_links_summary(monkeypatch, tmp_path) -> None:
    import json
    import torch
    from torch import nn

    from src.weather_net.config import AppConfig
    from src.weather_net.data import ManifestRow
    from src.weather_net.metrics import ClassificationReport
    from src.weather_net import training

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

    train_losses = [0.4, 0.3]
    eval_values = [(0.8, 0.51), (0.6, 0.73)]

    def fake_train_one_epoch(*_args, **_kwargs):
        return train_losses.pop(0)

    def fake_evaluate(*_args, **_kwargs):
        val_loss, macro_f1 = eval_values.pop(0)
        return (
            ClassificationReport(
                macro_f1=macro_f1,
                accuracy=macro_f1 + 0.1,
                per_class_f1={"rain": macro_f1, "sunny": macro_f1},
                confusion_matrix=[[1, 0], [0, 1]],
            ),
            val_loss,
        )

    monkeypatch.setattr(training, "load_training_manifest", fake_manifest)
    monkeypatch.setattr(training, "make_loaders", fake_loaders)
    monkeypatch.setattr(training, "create_classifier", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(
        training,
        "build_optimizer",
        lambda model, *_args, **_kwargs: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(training, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(training, "evaluate", fake_evaluate)
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
    config.train.epochs = 2
    config.train.output_dir = tmp_path / "outputs"

    training.train_config(config, device_request="cpu")

    summary = json.loads((config.train.output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert (config.train.output_dir / "training_history.csv").exists()
    assert (config.train.output_dir / "training_history.json").exists()
    assert (config.train.output_dir / "training_curves.png").exists()
    assert summary[0]["training_history_csv"] == str(config.train.output_dir / "training_history.csv")
    assert summary[0]["training_history_json"] == str(config.train.output_dir / "training_history.json")
    assert summary[0]["training_curves_png"] == str(config.train.output_dir / "training_curves.png")
    assert summary[0]["epochs"] == 2


def test_training_rejects_external_rows_without_merge_audit(monkeypatch, tmp_path) -> None:
    import pytest

    from src.weather_net.config import AppConfig
    from src.weather_net.data import ManifestRow
    from src.weather_net import training

    def fake_manifest(_config):
        rows = [
            ManifestRow(path=tmp_path / "rain.jpg", label=0, label_name="rain", image_id="rain.jpg"),
            ManifestRow(
                path=tmp_path / "external.jpg",
                label=0,
                label_name="rain",
                image_id="external.jpg",
                source="external_weapd",
                sample_weight=0.2,
            ),
        ]
        return rows, {"rain": 0}

    monkeypatch.setattr(training, "load_training_manifest", fake_manifest)
    config = AppConfig()
    config.data.train_csv = tmp_path / "train_with_external.csv"
    config.train.output_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="external merge audit"):
        training.train_config(config, device_request="cpu")


def test_training_accepts_external_rows_with_matching_merge_audit(monkeypatch, tmp_path) -> None:
    import torch
    from torch import nn
    from PIL import Image

    from src.weather_net.config import AppConfig
    from src.weather_net.data import build_manifest_from_csv
    from src.weather_net.metrics import ClassificationReport
    from src.weather_net import training
    from merge_external_training import merge_labeled_with_external_data

    def fake_manifest(_config):
        return build_manifest_from_csv(merged_csv)

    def fake_loaders(train_rows, val_rows, **_kwargs):
        return object(), object()

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(1, 1)

    monkeypatch.setattr(training, "load_training_manifest", fake_manifest)
    monkeypatch.setattr(training, "make_loaders", fake_loaders)
    monkeypatch.setattr(training, "create_classifier", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(training, "build_optimizer", lambda model, *_args, **_kwargs: torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(training, "train_one_epoch", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(
        training,
        "evaluate",
        lambda *_args, **_kwargs: (
            ClassificationReport(
                macro_f1=1.0,
                accuracy=1.0,
                per_class_f1={"rain": 1.0},
                confusion_matrix=[[1]],
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

    official_root = tmp_path / "official"
    external_root = tmp_path / "external"
    official_root.mkdir()
    external_root.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(official_root / "rain.jpg")
    Image.new("RGB", (8, 8), (11, 21, 31)).save(external_root / "rain_extra.jpg")
    train_csv = tmp_path / "official.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_weapd,0.2\n",
        encoding="utf-8",
    )
    merged_csv = tmp_path / "train_with_external.csv"
    audit_json = tmp_path / "audit.json"
    merge_labeled_with_external_data(
        train_csv=train_csv,
        train_dir=None,
        image_root=official_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=merged_csv,
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=audit_json,
    )
    config = AppConfig()
    config.data.train_csv = merged_csv
    config.data.external_audit_json = audit_json
    config.data.folds = 1
    config.train.epochs = 1
    config.train.output_dir = tmp_path / "outputs"

    training.train_config(config, device_request="cpu")

    assert (config.train.output_dir / "training_summary.json").exists()


def test_external_merge_audit_matches_relative_train_csv_from_cwd(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    from PIL import Image

    from src.weather_net.config import AppConfig
    from src.weather_net.data import build_manifest_from_csv
    from src.weather_net.training import validate_external_merge_audit
    from merge_external_training import merge_labeled_with_external_data

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "official").mkdir()
    (tmp_path / "external").mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "official" / "rain.jpg")
    Image.new("RGB", (8, 8), (11, 21, 31)).save(tmp_path / "external" / "rain_extra.jpg")
    Path("official.csv").write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    Path("external.csv").write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )
    train_csv = Path("data/train_with_external.csv")
    audit_json = Path("outputs/audit.json")
    merge_labeled_with_external_data(
        train_csv=Path("official.csv"),
        train_dir=None,
        image_root=Path("official"),
        external_csv=Path("external.csv"),
        external_image_root=Path("external"),
        output_csv=train_csv,
        rejected_csv=Path("outputs/rejected.csv"),
        audit_json=audit_json,
    )
    rows, _class_to_idx = build_manifest_from_csv(train_csv)
    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.external_audit_json = audit_json

    result = validate_external_merge_audit(rows, config)

    assert result is not None
    assert result["kept_external_rows"] == 1


def test_external_merge_audit_rejects_stale_source_counts(tmp_path) -> None:
    import json
    import pytest

    from src.weather_net.config import AppConfig
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import validate_external_merge_audit

    train_csv = tmp_path / "train_with_external.csv"
    audit_json = tmp_path / "audit.json"
    audit_json.write_text(
        json.dumps(
            {
                "output_csv": str(train_csv),
                "kept_external_rows": 1,
                "source_counts": {"labeled": 1, "external_weapd": 2},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        ManifestRow(path=tmp_path / "rain.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "external.jpg", label=0, label_name="rain", source="external_weapd"),
    ]
    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.external_audit_json = audit_json

    with pytest.raises(ValueError, match="source_counts"):
        validate_external_merge_audit(rows, config)


def test_external_merge_audit_rejects_missing_row_digest(tmp_path) -> None:
    import json
    import pytest
    from PIL import Image

    from src.weather_net.config import AppConfig
    from src.weather_net.data import build_manifest_from_csv
    from src.weather_net.training import validate_external_merge_audit

    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "rain.jpg")
    Image.new("RGB", (8, 8), (11, 21, 31)).save(tmp_path / "external.jpg")
    train_csv = tmp_path / "train_with_external.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain.jpg,rain,labeled,1.0\n"
        "external.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )
    audit_json = tmp_path / "audit.json"
    audit_json.write_text(
        json.dumps(
            {
                "output_csv": str(train_csv),
                "kept_external_rows": 1,
                "source_counts": {"labeled": 1, "external_weapd": 1},
            }
        ),
        encoding="utf-8",
    )
    rows, _class_to_idx = build_manifest_from_csv(train_csv)
    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.external_audit_json = audit_json

    with pytest.raises(ValueError, match="row_digest_sha256"):
        validate_external_merge_audit(rows, config)


def test_external_merge_audit_rejects_extra_stale_sources(tmp_path) -> None:
    import json
    import pytest

    from src.weather_net.config import AppConfig
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import validate_external_merge_audit

    train_csv = tmp_path / "train_with_external.csv"
    audit_json = tmp_path / "audit.json"
    audit_json.write_text(
        json.dumps(
            {
                "output_csv": str(train_csv),
                "kept_external_rows": 1,
                "source_counts": {"labeled": 1, "external_weapd": 1, "external_mwd": 10},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        ManifestRow(path=tmp_path / "rain.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "external.jpg", label=0, label_name="rain", source="external_weapd"),
    ]
    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.external_audit_json = audit_json

    with pytest.raises(ValueError, match="source_counts"):
        validate_external_merge_audit(rows, config)


def test_external_merge_audit_rejects_csv_content_changed_after_merge(tmp_path) -> None:
    import csv
    import pytest
    from PIL import Image

    from merge_external_training import merge_labeled_with_external_data
    from src.weather_net.config import AppConfig
    from src.weather_net.data import build_manifest_from_csv
    from src.weather_net.training import validate_external_merge_audit

    official_root = tmp_path / "official"
    external_root = tmp_path / "external"
    official_root.mkdir()
    external_root.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(official_root / "rain.jpg")
    Image.new("RGB", (8, 8), (11, 21, 31)).save(external_root / "rain_extra.jpg")
    train_csv = tmp_path / "official.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )
    merged_csv = tmp_path / "merged.csv"
    audit_json = tmp_path / "audit.json"
    merge_labeled_with_external_data(
        train_csv=train_csv,
        train_dir=None,
        image_root=official_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=merged_csv,
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=audit_json,
    )

    rows = list(csv.DictReader(merged_csv.open("r", encoding="utf-8")))
    rows[1]["sample_weight"] = "1.000000"
    with merged_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest, _class_to_idx = build_manifest_from_csv(merged_csv)
    config = AppConfig()
    config.data.train_csv = merged_csv
    config.data.external_audit_json = audit_json

    with pytest.raises(ValueError, match="row_digest"):
        validate_external_merge_audit(manifest, config)


def test_load_training_manifest_uses_official_class_map_for_csv(tmp_path) -> None:
    import json

    from src.weather_net.config import AppConfig
    from src.weather_net.training import load_training_manifest

    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "sunny.jpg").write_bytes(b"not-read-by-loader")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nsunny.jpg,sunny\n", encoding="utf-8")
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}) + "\n", encoding="utf-8")

    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.image_root = image_root
    config.data.class_map = class_map

    rows, class_to_idx = load_training_manifest(config)

    assert class_to_idx == {"rain": 0, "sunny": 1}
    assert rows[0].label == 1


def test_load_training_manifest_requires_class_map_for_csv(tmp_path) -> None:
    import pytest

    from src.weather_net.config import AppConfig
    from src.weather_net.training import load_training_manifest

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nsunny.jpg,sunny\n", encoding="utf-8")

    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.image_root = tmp_path

    with pytest.raises(ValueError, match="data.class_map"):
        load_training_manifest(config)


def test_load_training_manifest_rejects_csv_label_outside_official_class_map(tmp_path) -> None:
    import json
    import pytest

    from src.weather_net.config import AppConfig
    from src.weather_net.training import load_training_manifest

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrainbow.jpg,rainbow\n", encoding="utf-8")
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}) + "\n", encoding="utf-8")

    config = AppConfig()
    config.data.train_csv = train_csv
    config.data.image_root = tmp_path
    config.data.class_map = class_map

    with pytest.raises(ValueError, match="class mapping"):
        load_training_manifest(config)


def test_load_training_manifest_uses_official_class_map_for_imagefolder(tmp_path) -> None:
    import json

    from src.weather_net.config import AppConfig
    from src.weather_net.training import load_training_manifest

    train_dir = tmp_path / "train"
    (train_dir / "sunny").mkdir(parents=True)
    (train_dir / "sunny" / "sunny.jpg").write_bytes(b"not-read-by-loader")
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}) + "\n", encoding="utf-8")

    config = AppConfig()
    config.data.train_dir = train_dir
    config.data.class_map = class_map

    rows, class_to_idx = load_training_manifest(config)

    assert class_to_idx == {"rain": 0, "sunny": 1}
    assert rows[0].label == 1
