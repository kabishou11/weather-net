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


def test_load_config_accepts_macro_f1_loss_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_name: class_balanced_focal\n"
        "  focal_gamma: 1.5\n"
        "  class_balanced_beta: 0.99\n"
        "  sampler_mode: weighted\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.loss_name == "class_balanced_focal"
    assert config.train.focal_gamma == 1.5
    assert config.train.class_balanced_beta == 0.99
    assert config.train.sampler_mode == "weighted"


def test_load_config_accepts_inference_amp_option(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "infer:\n"
        "  amp: true\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.infer.amp is True


def test_load_config_rejects_invalid_loss_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_name: dice\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "loss_name" in str(error)
    else:
        raise AssertionError("invalid loss_name should fail during config load")


def test_load_config_rejects_invalid_class_balanced_beta(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  class_balanced_beta: 1.0\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "class_balanced_beta" in str(error)
    else:
        raise AssertionError("invalid class_balanced_beta should fail during config load")


def test_convnextv2_384_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384.yaml"))

    assert config.model.image_size == 384
    assert config.data.folds == 5
    assert config.train.loss_name == "class_balanced_focal"
