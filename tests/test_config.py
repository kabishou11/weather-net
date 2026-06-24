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


def test_load_config_accepts_optimizer_group_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  no_weight_decay: true\n"
        "  layer_decay: 0.75\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.75


def test_load_config_accepts_sample_weighted_sampler_mode(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  sampler_mode: sample_weighted\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.sampler_mode == "sample_weighted"


def test_load_config_accepts_sample_weight_usage(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  sample_weight_usage: sampler\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.sample_weight_usage == "sampler"


def test_load_config_accepts_balanced_softmax_loss(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_name: balanced_softmax\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.loss_name == "balanced_softmax"


def test_load_config_accepts_ldam_loss_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_name: ldam\n"
        "  ldam_max_margin: 0.4\n"
        "  ldam_scale: 20.0\n"
        "  loss_warmup_epochs: 2\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.loss_name == "ldam"
    assert config.train.ldam_max_margin == 0.4
    assert config.train.ldam_scale == 20.0
    assert config.train.loss_warmup_epochs == 2


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


def test_load_config_accepts_augmix_jsd_training_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  augment_policy: augmix_jsd\n"
        "train:\n"
        "  jsd_weight: 12.0\n"
        "  mixup_alpha: 0.0\n"
        "  cutmix_alpha: 0.0\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.augment_policy == "augmix_jsd"
    assert config.train.jsd_weight == 12.0


def test_load_config_rejects_augmix_jsd_without_consistency_weight(tmp_path: Path) -> None:
    import pytest
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  augment_policy: augmix_jsd\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="jsd_weight"):
        load_config(config_path)


def test_load_config_rejects_augmix_jsd_combined_with_mixup(tmp_path: Path) -> None:
    import pytest
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  augment_policy: augmix_jsd\n"
        "train:\n"
        "  jsd_weight: 12.0\n"
        "  mixup_alpha: 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MixUp/CutMix"):
        load_config(config_path)


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


def test_load_config_rejects_invalid_layer_decay(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  layer_decay: 0.0\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "layer_decay" in str(error)
    else:
        raise AssertionError("invalid layer_decay should fail during config load")


def test_load_config_rejects_invalid_ldam_options(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_name: ldam\n"
        "  ldam_max_margin: 0.0\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "ldam_max_margin" in str(error)
    else:
        raise AssertionError("invalid ldam_max_margin should fail during config load")


def test_load_config_rejects_negative_loss_warmup_epochs(tmp_path: Path) -> None:
    from src.weather_net.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  loss_warmup_epochs: -1\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "loss_warmup_epochs" in str(error)
    else:
        raise AssertionError("negative loss_warmup_epochs should fail during config load")


def test_convnextv2_384_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384.yaml"))

    assert config.model.image_size == 384
    assert config.data.folds == 5
    assert config.train.loss_name == "class_balanced_focal"
    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.85


def test_convnextv2_384_balanced_softmax_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384_balanced_softmax.yaml"))

    assert config.model.image_size == 384
    assert config.data.folds == 5
    assert config.train.loss_name == "balanced_softmax"
    assert config.train.sampler_mode == "auto"
    assert config.train.focal_gamma == 0.0
    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.85


def test_convnextv2_384_ldam_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384_ldam.yaml"))

    assert config.model.image_size == 384
    assert config.data.folds == 5
    assert config.train.loss_name == "ldam"
    assert config.train.ldam_max_margin == 0.5
    assert config.train.ldam_scale == 30.0
    assert config.train.loss_warmup_epochs == 2
    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.85


def test_convnextv2_384_augmix_jsd_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384_augmix_jsd.yaml"))

    assert config.model.image_size == 384
    assert config.data.augment_policy == "augmix_jsd"
    assert config.train.jsd_weight == 12.0
    assert config.train.mixup_alpha == 0.0
    assert config.train.cutmix_alpha == 0.0
    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.85


def test_convnextv2_384_hard_finetune_config_is_parseable() -> None:
    from src.weather_net.config import load_config

    config = load_config(Path("configs/convnextv2_384_hard_finetune.yaml"))

    assert config.model.image_size == 384
    assert config.train.sampler_mode == "sample_weighted"
    assert config.train.lr < 0.0002
    assert config.train.epochs == 8
    assert config.train.no_weight_decay is True
    assert config.train.layer_decay == 0.85
