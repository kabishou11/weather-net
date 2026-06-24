import json
import sys
from pathlib import Path

import numpy as np


def test_oof_decide_parse_args_accepts_multiple_oof_inputs(monkeypatch) -> None:
    from oof_decide import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oof_decide.py",
            "--oof",
            "a.npz",
            "b.npz",
            "--checkpoint",
            "a.pt",
            "b.pt",
            "--output-dir",
            "outputs/decision/exp001",
        ],
    )

    args = parse_args()

    assert args.oof == [Path("a.npz"), Path("b.npz")]
    assert args.checkpoint == [Path("a.pt"), Path("b.pt")]
    assert args.output_dir == Path("outputs/decision/exp001")


def test_oof_decide_writes_decision_params_from_oof_npz(tmp_path: Path) -> None:
    from oof_decide import run_oof_decision

    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.5, 0.4]], dtype=np.float32)
    npz_path = tmp_path / "oof_probabilities.npz"
    np.savez_compressed(
        npz_path,
        logits=logits,
        y_true=np.array([0, 1, 1], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
        image_id=np.array(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
    )

    stats = run_oof_decision(
        oof_paths=[npz_path],
        output_dir=tmp_path / "decision",
        temperature_candidates=[1.0],
        checkpoints=[Path("model.pt")],
    )

    decision_path = tmp_path / "decision" / "decision_params.json"
    report_path = tmp_path / "decision" / "decision_report.md"
    assert decision_path.exists()
    assert report_path.exists()
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert payload["class_names"] == ["rain", "sunny"]
    assert payload["weights"] == [1.0]
    assert payload["checkpoints"] == ["model.pt"]
    assert stats["num_models"] == 1


def test_oof_decide_expands_single_oof_weight_across_fold_checkpoints(tmp_path: Path) -> None:
    from oof_decide import run_oof_decision

    npz_path = tmp_path / "oof_probabilities.npz"
    np.savez_compressed(
        npz_path,
        logits=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        y_true=np.array([0, 1], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
        image_id=np.array(["fold0/a.jpg", "fold1/b.jpg"], dtype=object),
        checkpoint=np.array(["fold0.pt", "fold1.pt"], dtype=object),
    )

    stats = run_oof_decision(
        oof_paths=[npz_path],
        output_dir=tmp_path / "decision",
        temperature_candidates=[1.0],
        checkpoints=[Path("fold0.pt"), Path("fold1.pt")],
    )

    payload = json.loads((tmp_path / "decision" / "decision_params.json").read_text(encoding="utf-8"))

    assert payload["weights"] == [0.5, 0.5]
    assert payload["checkpoints"] == ["fold0.pt", "fold1.pt"]
    assert stats["weights"] == [0.5, 0.5]


def test_oof_decide_rejects_mixed_checkpoint_metadata(tmp_path: Path) -> None:
    import pytest
    from oof_decide import run_oof_decision

    first = tmp_path / "with_checkpoint.npz"
    second = tmp_path / "without_checkpoint.npz"
    common_payload = {
        "logits": np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        "y_true": np.array([0, 1], dtype=np.int64),
        "class_names": np.array(["rain", "sunny"], dtype=object),
        "image_id": np.array(["a.jpg", "b.jpg"], dtype=object),
    }
    np.savez_compressed(first, **common_payload, checkpoint=np.array(["fold0.pt", "fold1.pt"], dtype=object))
    np.savez_compressed(second, **common_payload)

    with pytest.raises(ValueError, match="checkpoint metadata"):
        run_oof_decision(
            oof_paths=[first, second],
            output_dir=tmp_path / "decision",
            temperature_candidates=[1.0],
            checkpoints=[Path("fold0.pt"), Path("fold1.pt")],
        )


def test_oof_decide_rejects_oof_without_image_ids(tmp_path: Path) -> None:
    import pytest
    from oof_decide import run_oof_decision

    npz_path = tmp_path / "bad_oof.npz"
    np.savez_compressed(
        npz_path,
        logits=np.array([[2.0, 0.0]], dtype=np.float32),
        y_true=np.array([0], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
    )

    with pytest.raises(ValueError, match="image_id"):
        run_oof_decision(
            oof_paths=[npz_path],
            output_dir=tmp_path / "decision",
            temperature_candidates=[1.0],
        )


def test_oof_decide_rejects_mismatched_oof_lengths(tmp_path: Path) -> None:
    import pytest
    from oof_decide import run_oof_decision

    npz_path = tmp_path / "bad_lengths.npz"
    np.savez_compressed(
        npz_path,
        logits=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        y_true=np.array([0], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
        image_id=np.array(["a.jpg", "b.jpg"], dtype=object),
    )

    with pytest.raises(ValueError, match="same number of rows"):
        run_oof_decision(
            oof_paths=[npz_path],
            output_dir=tmp_path / "decision",
            temperature_candidates=[1.0],
        )


def test_oof_decide_rejects_duplicate_image_ids_within_oof(tmp_path: Path) -> None:
    import pytest
    from oof_decide import run_oof_decision

    npz_path = tmp_path / "duplicate_ids.npz"
    np.savez_compressed(
        npz_path,
        logits=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        y_true=np.array([0, 1], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
        image_id=np.array(["same.jpg", "same.jpg"], dtype=object),
    )

    with pytest.raises(ValueError, match="Duplicate OOF image ids"):
        run_oof_decision(
            oof_paths=[npz_path],
            output_dir=tmp_path / "decision",
            temperature_candidates=[1.0],
        )


def test_oof_decide_rejects_logits_class_dimension_mismatch(tmp_path: Path) -> None:
    import pytest
    from oof_decide import run_oof_decision

    npz_path = tmp_path / "bad_classes.npz"
    np.savez_compressed(
        npz_path,
        logits=np.array([[2.0, 0.0, -1.0]], dtype=np.float32),
        y_true=np.array([0], dtype=np.int64),
        class_names=np.array(["rain", "sunny"], dtype=object),
        image_id=np.array(["a.jpg"], dtype=object),
    )

    with pytest.raises(ValueError, match="class dimension"):
        run_oof_decision(
            oof_paths=[npz_path],
            output_dir=tmp_path / "decision",
            temperature_candidates=[1.0],
        )
