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


def test_apply_tau_norm_supports_nested_head_keys() -> None:
    from classifier_rebalance import apply_tau_norm_to_state_dict

    state = {"head.fc.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]])}

    output = apply_tau_norm_to_state_dict(state, tau=1.0, head_key="auto")

    assert torch.allclose(output["head.fc.weight"], torch.tensor([[0.6, 0.8], [0.0, 1.0]]), atol=1e-6)


def test_fold_lws_scales_into_classifier_rows_without_changing_other_tensors() -> None:
    from classifier_rebalance import fold_lws_into_state_dict

    state = {
        "backbone.weight": torch.ones(2, 2),
        "classifier.weight": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        "classifier.bias": torch.tensor([0.5, -0.5]),
    }
    log_scales = torch.log(torch.tensor([0.5, 2.0]))

    output = fold_lws_into_state_dict(state, log_scales=log_scales, head_key="auto")

    assert torch.allclose(output["classifier.weight"], torch.tensor([[1.5, 2.0], [0.0, 4.0]]))
    assert torch.allclose(output["classifier.bias"], torch.tensor([0.25, -1.0]))
    assert torch.equal(output["backbone.weight"], state["backbone.weight"])
    assert torch.allclose(
        torch.nn.functional.normalize(output["classifier.weight"], dim=1),
        torch.nn.functional.normalize(state["classifier.weight"], dim=1),
        atol=1e-6,
    )


def test_fold_lws_rejects_non_finite_or_extreme_scales() -> None:
    import pytest

    from classifier_rebalance import fold_lws_into_state_dict

    state = {"classifier.weight": torch.ones(2, 2)}

    with pytest.raises(ValueError, match="finite"):
        fold_lws_into_state_dict(state, log_scales=torch.tensor([0.0, float("inf")]))
    with pytest.raises(ValueError, match="range"):
        fold_lws_into_state_dict(state, log_scales=torch.log(torch.tensor([0.1, 1.0])), min_scale=0.25)
    with pytest.raises(ValueError, match="range"):
        fold_lws_into_state_dict(state, log_scales=torch.log(torch.tensor([1.0, 5.0])), max_scale=4.0)


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


def test_parse_args_accepts_lws_cli(monkeypatch) -> None:
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
            "fold0_lws.pt",
            "--method",
            "lws",
            "--train-csv",
            "train.csv",
            "--image-root",
            "images",
            "--epochs",
            "3",
            "--sampler-mode",
            "class_balanced",
        ],
    )

    args = parse_args()

    assert args.method == "lws"
    assert args.train_csv == Path("train.csv")
    assert args.image_root == Path("images")
    assert args.epochs == 3
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


def test_retrain_classifier_head_preserves_explicit_head_prefix(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr(classifier_rebalance, "create_classifier", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(classifier_rebalance, "build_crt_loader", lambda *_args, **_kwargs: loader)
    monkeypatch.setattr(classifier_rebalance, "load_training_manifest_from_args", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(classifier_rebalance, "train_one_epoch", lambda *_args, **_kwargs: 0.1)

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
        head_prefix="classifier",
        device="cpu",
        num_workers=0,
    )

    assert summary["head_key"] == "classifier.weight"


def test_retrain_classifier_head_rejects_invalid_class_mapping(monkeypatch, tmp_path: Path) -> None:
    import pytest
    from torch import nn

    import classifier_rebalance

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(2, 2)

        def forward(self, x):
            return self.classifier(x)

    checkpoint = {
        "model_state": TinyModel().state_dict(),
        "class_to_idx": {"rain": 0, "sunny": 0},
        "model_name": "tiny",
        "image_size": 8,
    }
    checkpoint_path = tmp_path / "input.pt"
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(classifier_rebalance, "create_classifier", lambda *_args, **_kwargs: TinyModel())

    with pytest.raises(ValueError, match="contiguous"):
        classifier_rebalance.retrain_classifier_head(
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "crt.pt",
            train_csv=tmp_path / "train.csv",
            train_dir=None,
            device="cpu",
            num_workers=0,
        )


def test_train_lws_rebalance_learns_only_class_scales_and_folds_them(monkeypatch, tmp_path: Path) -> None:
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

    model = TinyModel()
    model.classifier.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    model.classifier.bias.data = torch.tensor([0.1, -0.2])
    checkpoint = {
        "model_state": model.state_dict(),
        "class_to_idx": {"rain": 0, "sunny": 1},
        "model_name": "tiny",
        "image_size": 8,
    }
    checkpoint_path = tmp_path / "input.pt"
    output_path = tmp_path / "lws.pt"
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

    def fake_train_lws_one_epoch(model, loader, optimizer, device, lws_log_scales):
        seen["trainable"] = [name for name, param in model.named_parameters() if param.requires_grad]
        seen["scale_requires_grad"] = lws_log_scales.requires_grad
        with torch.no_grad():
            lws_log_scales.copy_(torch.log(torch.tensor([0.5, 2.0], device=lws_log_scales.device)))
        return 0.2

    monkeypatch.setattr(classifier_rebalance, "train_lws_one_epoch", fake_train_lws_one_epoch)

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
        rebalance_method="lws",
    )
    output = torch.load(output_path, map_location="cpu")

    assert seen["trainable"] == []
    assert seen["scale_requires_grad"] is True
    assert summary["method"] == "lws"
    assert output["rebalance"]["scales_folded_into"] == "weight_and_bias"
    assert output["rebalance"]["scale_by_class"] == {"rain": 0.5, "sunny": 2.0}
    assert output["rebalance"]["method"] == "lws"
    assert torch.allclose(output["model_state"]["classifier.weight"], torch.tensor([[0.5, 0.0], [0.0, 2.0]]))
    assert torch.allclose(output["model_state"]["classifier.bias"], torch.tensor([0.05, -0.4]))


def test_train_lws_one_epoch_updates_only_scales() -> None:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from classifier_rebalance import freeze_all_model_parameters, train_lws_one_epoch

    model = nn.Linear(2, 2)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    freeze_all_model_parameters(model)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([0, 1]),
            torch.ones(2),
        ),
        batch_size=2,
    )
    scales = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([scales], lr=0.1)

    loss = train_lws_one_epoch(model, loader, optimizer, device="cpu", lws_log_scales=scales)

    assert loss > 0
    assert scales.grad is not None
    assert not torch.allclose(scales.detach(), torch.zeros(2))
    for name, param in model.named_parameters():
        assert param.requires_grad is False
        assert torch.equal(param.detach(), before[name])


def test_train_lws_one_epoch_accepts_weather_dataset_four_tuple_batches() -> None:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from classifier_rebalance import freeze_all_model_parameters, train_lws_one_epoch

    model = nn.Linear(2, 2)
    freeze_all_model_parameters(model)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([0, 1]),
            torch.ones(2),
            torch.empty(2, 0),
        ),
        batch_size=2,
    )
    scales = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([scales], lr=0.1)

    loss = train_lws_one_epoch(model, loader, optimizer, device="cpu", lws_log_scales=scales)

    assert loss > 0
    assert scales.grad is not None
