from pathlib import Path

import torch


def test_apply_tau_norm_scales_classifier_rows_without_changing_directions() -> None:
    from classifier_rebalance import apply_tau_norm_to_state_dict

    state = {
        "backbone.weight": torch.ones(2, 2),
        "classifier.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        "classifier.bias": torch.tensor([0.5, -0.5]),
    }

    output = apply_tau_norm_to_state_dict(state, tau=1.0, head_key="auto")

    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    assert torch.allclose(output["classifier.weight"], expected, atol=1e-6)
    assert torch.equal(output["classifier.bias"], state["classifier.bias"])
    assert torch.equal(output["backbone.weight"], state["backbone.weight"])
    assert torch.allclose(
        torch.nn.functional.normalize(output["classifier.weight"], dim=1),
        torch.nn.functional.normalize(state["classifier.weight"], dim=1),
        atol=1e-6,
    )


def test_apply_tau_norm_supports_partial_tau_and_preserves_missing_head_error() -> None:
    import pytest

    from classifier_rebalance import apply_tau_norm_to_state_dict

    state = {"head.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]])}

    output = apply_tau_norm_to_state_dict(state, tau=0.5, head_key="head.weight")

    expected = state["head.weight"] / torch.tensor([[5.0], [2.0]]).sqrt()
    assert torch.allclose(output["head.weight"], expected, atol=1e-6)
    with pytest.raises(ValueError, match="classifier head"):
        apply_tau_norm_to_state_dict({"backbone.weight": torch.ones(1)}, tau=1.0, head_key="auto")


def test_apply_tau_norm_rejects_ambiguous_classifier_heads() -> None:
    import pytest

    from classifier_rebalance import apply_tau_norm_to_state_dict

    state = {
        "classifier.weight": torch.ones(2, 3),
        "head.weight": torch.ones(2, 3),
    }

    with pytest.raises(ValueError, match="Multiple classifier head"):
        apply_tau_norm_to_state_dict(state, tau=1.0, head_key="auto")


def test_rebalance_checkpoint_writes_tau_norm_metadata(tmp_path: Path) -> None:
    from classifier_rebalance import rebalance_checkpoint

    checkpoint = {
        "model_state": {
            "classifier.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
            "classifier.bias": torch.zeros(2),
        },
        "class_to_idx": {"rain": 0, "sunny": 1},
        "model_name": "small_cnn",
        "image_size": 224,
    }
    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    torch.save(checkpoint, input_path)

    summary = rebalance_checkpoint(input_path, output_path, tau=1.0, head_key="auto", map_location="cpu")
    output = torch.load(output_path, map_location="cpu")

    assert summary["output"] == str(output_path)
    assert summary["method"] == "tau_norm"
    assert summary["head_key"] == "classifier.weight"
    assert output["rebalance"]["method"] == "tau_norm"
    assert output["rebalance"]["tau"] == 1.0
    assert torch.allclose(output["model_state"]["classifier.weight"].norm(dim=1), torch.ones(2), atol=1e-6)


def test_parse_args_accepts_tau_norm_cli(monkeypatch) -> None:
    import sys

    from classifier_rebalance import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classifier_rebalance.py",
            "--checkpoint",
            "fold0.pt",
            "--output",
            "fold0_tau.pt",
            "--tau",
            "0.75",
            "--head-key",
            "classifier.weight",
        ],
    )

    args = parse_args()

    assert args.checkpoint == Path("fold0.pt")
    assert args.output == Path("fold0_tau.pt")
    assert args.tau == 0.75
    assert args.head_key == "classifier.weight"


def test_parse_args_accepts_crt_cli(monkeypatch) -> None:
    import sys

    from classifier_rebalance import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classifier_rebalance.py",
            "--checkpoint",
            "fold0.pt",
            "--output",
            "fold0_crt.pt",
            "--method",
            "crt",
            "--train-csv",
            "train.csv",
            "--image-root",
            "images",
            "--epochs",
            "5",
            "--sampler-mode",
            "class_balanced",
        ],
    )

    args = parse_args()

    assert args.method == "crt"
    assert args.train_csv == Path("train.csv")
    assert args.image_root == Path("images")
    assert args.epochs == 5
    assert args.sampler_mode == "class_balanced"


def test_freeze_backbone_for_classifier_retraining_only_keeps_head_trainable() -> None:
    from torch import nn

    from classifier_rebalance import freeze_backbone_for_classifier_retraining

    model = nn.Sequential()
    model.backbone = nn.Linear(4, 4)
    model.classifier = nn.Linear(4, 2)

    trainable = freeze_backbone_for_classifier_retraining(model, head_prefix="classifier")

    assert trainable == ["classifier.weight", "classifier.bias"]
    assert model.backbone.weight.requires_grad is False
    assert model.backbone.bias.requires_grad is False
    assert model.classifier.weight.requires_grad is True
    assert model.classifier.bias.requires_grad is True


def test_build_classifier_balanced_sampler_supports_sqrt_weights() -> None:
    from src.weather_net.data import ManifestRow
    from classifier_rebalance import build_classifier_balanced_sampler

    rows = [
        ManifestRow(path=Path("a.jpg"), label=0),
        ManifestRow(path=Path("b.jpg"), label=0),
        ManifestRow(path=Path("c.jpg"), label=0),
        ManifestRow(path=Path("d.jpg"), label=1),
    ]

    sampler = build_classifier_balanced_sampler(rows, mode="sqrt")

    assert sampler is not None
    assert list(sampler.weights.tolist()) == [1 / 3**0.5, 1 / 3**0.5, 1 / 3**0.5, 1.0]


def test_build_classifier_balanced_sampler_rejects_unlabeled_rows() -> None:
    import pytest

    from src.weather_net.data import ManifestRow
    from classifier_rebalance import build_classifier_balanced_sampler

    with pytest.raises(ValueError, match="labeled"):
        build_classifier_balanced_sampler([ManifestRow(path=Path("a.jpg"), label=None)], mode="class_balanced")


def test_load_training_manifest_from_args_uses_checkpoint_class_mapping(tmp_path: Path) -> None:
    from classifier_rebalance import load_training_manifest_from_args

    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image,label\n"
        "sunny.jpg,sunny\n"
        "rain.jpg,rain\n",
        encoding="utf-8",
    )

    rows = load_training_manifest_from_args(
        train_csv=csv_path,
        train_dir=None,
        image_root=tmp_path,
        class_to_idx={"sunny": 0, "rain": 1},
    )

    assert [(row.label_name, row.label) for row in rows] == [("sunny", 0), ("rain", 1)]


def test_retrain_classifier_head_freezes_backbone_and_writes_metadata(monkeypatch, tmp_path: Path) -> None:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import classifier_rebalance
    from src.weather_net.data import ManifestRow

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.classifier = nn.Linear(2, 2)

        def forward(self, x):
            return self.classifier(self.backbone(x))

    checkpoint = {
        "model_state": TinyModel().state_dict(),
        "class_to_idx": {"rain": 0, "sunny": 1},
        "model_name": "tiny",
        "image_size": 8,
    }
    checkpoint_path = tmp_path / "input.pt"
    output_path = tmp_path / "crt.pt"
    torch.save(checkpoint, checkpoint_path)
    rows = [
        ManifestRow(path=tmp_path / "rain.jpg", label=0, label_name="rain"),
        ManifestRow(path=tmp_path / "sunny.jpg", label=1, label_name="sunny"),
    ]
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.tensor([0, 1]), torch.ones(2)),
        batch_size=2,
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(classifier_rebalance, "create_classifier", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(classifier_rebalance, "build_crt_loader", lambda *_args, **_kwargs: loader)
    monkeypatch.setattr(classifier_rebalance, "load_training_manifest_from_args", lambda *_args, **_kwargs: rows)

    def fake_train_one_epoch(model, *_args, **_kwargs):
        seen["trainable"] = [name for name, param in model.named_parameters() if param.requires_grad]
        return 0.1

    monkeypatch.setattr(classifier_rebalance, "train_one_epoch", fake_train_one_epoch)

    summary = classifier_rebalance.retrain_classifier_head(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        train_csv=tmp_path / "train.csv",
        train_dir=None,
        image_root=None,
        epochs=1,
        batch_size=2,
        lr=1e-2,
        sampler_mode="sqrt",
        device="cpu",
        num_workers=0,
    )
    output = torch.load(output_path, map_location="cpu")

    assert seen["trainable"] == ["classifier.weight", "classifier.bias"]
    assert summary["method"] == "crt"
    assert summary["sampler_mode"] == "sqrt"
    assert output["rebalance"]["method"] == "crt"
    assert output["rebalance"]["epochs"] == 1
