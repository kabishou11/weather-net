import json
import sys
from pathlib import Path


def test_pseudo_label_parse_args_accepts_class_aware_files(monkeypatch) -> None:
    from pseudo_label import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pseudo_label.py",
            "--checkpoint",
            "model.pt",
            "--per-class-thresholds",
            "thresholds.json",
            "--per-class-max-count",
            "caps.json",
        ],
    )

    args = parse_args()

    assert args.per_class_thresholds == Path("thresholds.json")
    assert args.per_class_max_count == Path("caps.json")


def test_pseudo_label_parse_args_allows_oof_threshold_mode_without_checkpoint(monkeypatch) -> None:
    from pseudo_label import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pseudo_label.py",
            "--estimate-thresholds-from-oof",
            "oof.npz",
            "--threshold-output",
            "thresholds.json",
        ],
    )

    args = parse_args()

    assert args.checkpoint is None
    assert args.estimate_thresholds_from_oof == Path("oof.npz")
    assert args.threshold_output == Path("thresholds.json")


def test_pseudo_label_parse_args_requires_checkpoint_for_label_generation(monkeypatch) -> None:
    import pytest
    from pseudo_label import parse_args

    monkeypatch.setattr(sys, "argv", ["pseudo_label.py", "--test-dir", "images"])

    with pytest.raises(SystemExit):
        parse_args()


def test_load_class_float_mapping_reads_threshold_json(tmp_path: Path) -> None:
    from pseudo_label import load_class_float_mapping

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"rain": 0.91, "sunny": "0.95"}), encoding="utf-8")

    assert load_class_float_mapping(path) == {"rain": 0.91, "sunny": 0.95}


def test_load_class_int_mapping_reads_cap_json(tmp_path: Path) -> None:
    from pseudo_label import load_class_int_mapping

    path = tmp_path / "caps.json"
    path.write_text(json.dumps({"rain": 100, "sunny": "50"}), encoding="utf-8")

    assert load_class_int_mapping(path) == {"rain": 100, "sunny": 50}


def test_load_class_int_mapping_rejects_float_caps(tmp_path: Path) -> None:
    import pytest
    from pseudo_label import load_class_int_mapping

    path = tmp_path / "caps.json"
    path.write_text(json.dumps({"rain": 10.9}), encoding="utf-8")

    with pytest.raises(ValueError, match="integer"):
        load_class_int_mapping(path)


def test_load_class_int_mapping_rejects_bool_caps(tmp_path: Path) -> None:
    import pytest
    from pseudo_label import load_class_int_mapping

    path = tmp_path / "caps.json"
    path.write_text(json.dumps({"rain": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="integer"):
        load_class_int_mapping(path)


def test_generate_thresholds_from_oof_npz(tmp_path: Path) -> None:
    import numpy as np
    from pseudo_label import generate_thresholds_from_oof

    oof_path = tmp_path / "oof.npz"
    output_path = tmp_path / "thresholds.json"
    np.savez_compressed(
        oof_path,
        y_true=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        probs=np.array(
            [
                [0.95, 0.05],
                [0.80, 0.20],
                [0.40, 0.60],
                [0.10, 0.90],
                [0.70, 0.30],
            ],
            dtype=np.float32,
        ),
        class_names=np.array(["rain", "sunny"], dtype=object),
    )

    thresholds = generate_thresholds_from_oof(
        oof_path=oof_path,
        output_path=output_path,
        target_precision=0.75,
        min_threshold=0.5,
        fallback_threshold=0.97,
    )

    assert thresholds == {"rain": 0.8, "sunny": 0.6}
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"rain": 0.8, "sunny": 0.6}


def test_pseudo_label_main_generates_thresholds_without_checkpoint(monkeypatch, tmp_path: Path, capsys) -> None:
    import numpy as np
    from pseudo_label import main

    oof_path = tmp_path / "oof.npz"
    output_path = tmp_path / "thresholds.json"
    np.savez_compressed(
        oof_path,
        y_true=np.array([0, 1], dtype=np.int64),
        probs=np.array([[0.95, 0.05], [0.10, 0.90]], dtype=np.float32),
        class_names=np.array(["rain", "sunny"], dtype=object),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pseudo_label.py",
            "--estimate-thresholds-from-oof",
            str(oof_path),
            "--threshold-output",
            str(output_path),
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"rain": 0.95, "sunny": 0.9}


def test_generate_thresholds_from_oof_rejects_mismatched_rows(tmp_path: Path) -> None:
    import numpy as np
    import pytest
    from pseudo_label import generate_thresholds_from_oof

    oof_path = tmp_path / "bad_oof.npz"
    np.savez_compressed(
        oof_path,
        y_true=np.array([0], dtype=np.int64),
        probs=np.array([[0.95, 0.05], [0.10, 0.90]], dtype=np.float32),
        class_names=np.array(["rain", "sunny"], dtype=object),
    )

    with pytest.raises(ValueError, match="same number of rows"):
        generate_thresholds_from_oof(
            oof_path=oof_path,
            output_path=tmp_path / "thresholds.json",
            target_precision=0.95,
            min_threshold=0.8,
            fallback_threshold=0.99,
        )


def test_generate_thresholds_from_oof_rejects_duplicate_class_names(tmp_path: Path) -> None:
    import numpy as np
    import pytest
    from pseudo_label import generate_thresholds_from_oof

    oof_path = tmp_path / "bad_oof.npz"
    np.savez_compressed(
        oof_path,
        y_true=np.array([0, 1], dtype=np.int64),
        probs=np.array([[0.95, 0.05], [0.10, 0.90]], dtype=np.float32),
        class_names=np.array(["rain", "rain"], dtype=object),
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        generate_thresholds_from_oof(
            oof_path=oof_path,
            output_path=tmp_path / "thresholds.json",
            target_precision=0.95,
            min_threshold=0.8,
            fallback_threshold=0.99,
        )
