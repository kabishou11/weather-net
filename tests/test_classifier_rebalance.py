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
