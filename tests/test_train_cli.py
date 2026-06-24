from pathlib import Path

import pytest


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        "train:\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "data:\n"
        "  folds: 1\n",
        encoding="utf-8",
    )


def test_train_parse_args_accepts_class_map(monkeypatch) -> None:
    from train import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-csv",
            "data/train.csv",
            "--image-root",
            "data/images",
            "--class-map",
            "outputs/convnextv2_384/class_to_idx.json",
        ],
    )

    args = parse_args()

    assert args.class_map == Path("outputs/convnextv2_384/class_to_idx.json")


def test_train_parse_args_accepts_preflight_gates(monkeypatch) -> None:
    from train import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-csv",
            "data/train.csv",
            "--image-root",
            "data/images",
            "--class-map",
            "outputs/class_to_idx.json",
            "--preflight",
            "server-strict",
            "--allow-external-data",
            "--external-max-ratio",
            "0.3",
            "--external-max-sample-weight",
            "0.5",
            "--pseudo-min-confidence",
            "0.95",
            "--pseudo-max-ratio",
            "0.5",
            "--allow-pseudo-teacher-distillation",
            "--inference-stats",
            "outputs/submission.stats.json",
            "--max-checkpoints",
            "1",
        ],
    )

    args = parse_args()

    assert args.preflight == "server-strict"
    assert args.allow_external_data is True
    assert args.external_max_ratio == 0.3
    assert args.external_max_sample_weight == 0.5
    assert args.pseudo_min_confidence == 0.95
    assert args.pseudo_max_ratio == 0.5
    assert args.allow_pseudo_teacher_distillation is True
    assert args.inference_stats == Path("outputs/submission.stats.json")
    assert args.max_checkpoints == 1


def test_train_parse_args_defaults_to_server_strict(monkeypatch) -> None:
    from train import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-csv",
            "data/train.csv",
            "--image-root",
            "data/images",
            "--class-map",
            "outputs/class_to_idx.json",
        ],
    )

    args = parse_args()

    assert args.preflight == "server-strict"


def test_train_main_rejects_preflight_off_without_confirmation(monkeypatch, tmp_path: Path) -> None:
    import train

    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--preflight",
            "off",
        ],
    )

    with pytest.raises(ValueError, match="--i-understand-preflight-off"):
        train.main()


def test_train_main_runs_preflight_before_training(monkeypatch, tmp_path: Path) -> None:
    import json

    import train

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)
    output_dir = tmp_path / "outputs"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--train-csv",
            str(train_csv),
            "--image-root",
            str(tmp_path),
            "--class-map",
            str(class_map),
            "--preflight",
            "server-strict",
            "--output-dir",
            str(output_dir),
        ],
    )

    def fake_preflight(**kwargs):
        calls.append(("preflight", kwargs))
        return {"status": "pass"}

    def fake_train_config(config, device_request="auto"):
        calls.append(("train", config.train.output_dir))
        return [tmp_path / "best.pt"]

    monkeypatch.setattr(train, "run_preflight_checks", fake_preflight)
    monkeypatch.setattr(train, "train_config", fake_train_config)

    train.main()

    assert calls[0][0] == "preflight"
    assert calls[0][1]["profile"] == "server-strict"
    assert calls[0][1]["class_map"] == class_map
    assert calls[0][1]["config"].data.class_map == class_map
    assert calls[0][1]["config"].train.output_dir == output_dir
    assert calls[1] == ("train", output_dir)


def test_train_main_preflight_uses_cli_folds_override(monkeypatch, tmp_path: Path) -> None:
    import json

    from PIL import Image

    import train

    samples = [
        ("rain1.jpg", "rain", (10, 20, 30)),
        ("rain2.jpg", "rain", (11, 20, 30)),
        ("sunny1.jpg", "sunny", (30, 20, 10)),
        ("sunny2.jpg", "sunny", (30, 21, 10)),
    ]
    image_root = tmp_path / "images"
    image_root.mkdir()
    for image_name, _label, color in samples:
        Image.new("RGB", (8, 8), color).save(image_root / image_name)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source\n" + "".join(f"{image_name},{label},labeled\n" for image_name, label, _color in samples),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--train-csv",
            str(train_csv),
            "--image-root",
            str(image_root),
            "--class-map",
            str(class_map),
            "--folds",
            "3",
        ],
    )

    monkeypatch.setattr(
        train,
        "train_config",
        lambda _config, device_request="auto": pytest.fail("train_config should not run after preflight failure"),
    )

    with pytest.raises(ValueError, match="labeled classes below requested folds"):
        train.main()


def test_train_main_rejects_csv_and_imagefolder_inputs(monkeypatch, tmp_path: Path) -> None:
    import train

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    train_dir = tmp_path / "train_dir"
    train_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--train-csv",
            str(train_csv),
            "--train-dir",
            str(train_dir),
        ],
    )

    with pytest.raises(ValueError, match="either --train-csv or --train-dir"):
        train.main()


def test_train_main_revalidates_cli_overrides(monkeypatch) -> None:
    import train
    from pathlib import Path

    config_path = Path("/tmp/weather_net_train_cli_config.yaml")
    _write_minimal_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--batch-size",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="train.batch_size"):
        train.main()


def test_train_main_rejects_missing_config(monkeypatch) -> None:
    import train

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "/tmp/weather_net_missing_config.yaml",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        train.main()
