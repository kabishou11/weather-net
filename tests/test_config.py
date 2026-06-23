from pathlib import Path


def test_load_config_coerces_path_fields_on_python39(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  train_dir: data/train\n"
        "train:\n"
        "  output_dir: outputs/run\n"
        "infer:\n"
        "  output_csv: submission.csv\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.train_dir == Path("data/train")
    assert config.train.output_dir == Path("outputs/run")
    assert config.infer.output_csv == Path("submission.csv")


def test_load_config_accepts_ema_decay(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  ema_decay: 0.999\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.ema_decay == 0.999
