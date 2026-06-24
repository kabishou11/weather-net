import json
import sys
from pathlib import Path

import numpy as np
import torch


def _checkpoint(weight: float) -> dict[str, object]:
    return {
        "model_state": {
            "linear.weight": torch.tensor([[weight, weight + 1.0]]),
            "linear.bias": torch.tensor([weight]),
        },
        "class_to_idx": {"rain": 0, "sunny": 1},
        "model_name": "small_cnn",
        "image_size": 64,
        "macro_f1": weight,
    }


def test_soup_checkpoints_parse_args_accepts_oof_gate(monkeypatch) -> None:
    from soup_checkpoints import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "soup_checkpoints.py",
            "--checkpoints",
            "a.pt",
            "b.pt",
            "--output",
            "soup.pt",
            "--oof",
            "a_oof.npz",
            "b_oof.npz",
            "--min-delta",
            "0.01",
        ],
    )

    args = parse_args()

    assert args.oof == [Path("a_oof.npz"), Path("b_oof.npz")]
    assert args.min_delta == 0.01


def test_soup_checkpoints_builds_oof_gated_soup_from_oof_npz(tmp_path: Path) -> None:
    from soup_checkpoints import run_soup

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_checkpoint(1.0), first)
    torch.save(_checkpoint(3.0), second)
    common = {
        "y_true": np.array([0, 1, 1, 0], dtype=np.int64),
        "class_names": np.array(["rain", "sunny"], dtype=object),
        "image_id": np.array(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object),
    }
    first_oof = tmp_path / "first_oof.npz"
    second_oof = tmp_path / "second_oof.npz"
    np.savez_compressed(
        first_oof,
        logits=np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], dtype=np.float32),
        **common,
    )
    np.savez_compressed(
        second_oof,
        logits=np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]], dtype=np.float32),
        **common,
    )
    output = tmp_path / "soup.pt"

    stats = run_soup(
        checkpoints=[first, second],
        output=output,
        weights=None,
        map_location="cpu",
        oof_paths=[first_oof, second_oof],
        min_delta=0.0,
        max_steps=4,
    )

    saved = torch.load(output, map_location="cpu")

    assert stats["soup"]["mode"] == "oof_gated_greedy"
    assert saved["soup"]["mode"] == "oof_gated_greedy"
    assert saved["soup"]["oof_score"] >= saved["soup"]["baseline_oof_score"]
    assert output.exists()


def test_soup_checkpoints_rejects_misaligned_oof_ids(tmp_path: Path) -> None:
    import pytest
    from soup_checkpoints import run_soup

    checkpoint_path = tmp_path / "model.pt"
    torch.save(_checkpoint(1.0), checkpoint_path)
    first_oof = tmp_path / "first_oof.npz"
    second_oof = tmp_path / "second_oof.npz"
    payload = {
        "logits": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        "y_true": np.array([0, 1], dtype=np.int64),
        "class_names": np.array(["rain", "sunny"], dtype=object),
    }
    np.savez_compressed(first_oof, **payload, image_id=np.array(["a.jpg", "b.jpg"], dtype=object))
    np.savez_compressed(second_oof, **payload, image_id=np.array(["b.jpg", "a.jpg"], dtype=object))

    with pytest.raises(ValueError, match="image_id order"):
        run_soup(
            checkpoints=[checkpoint_path, checkpoint_path],
            output=tmp_path / "soup.pt",
            weights=None,
            map_location="cpu",
            oof_paths=[first_oof, second_oof],
        )
