from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_dir: Path | None = None
    train_csv: Path | None = None
    image_root: Path | None = None
    test_dir: Path | None = None
    test_csv: Path | None = None
    image_column: str | None = None
    label_column: str | None = None
    val_fraction: float = 0.2
    folds: int = 1
    seed: int = 42
    num_workers: int = 2
    augment_policy: str = "standard"
    transform_backend: str = "auto"


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    image_size: int = 224
    pretrained: bool = True
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    no_weight_decay: bool = False
    layer_decay: float = 1.0
    label_smoothing: float = 0.05
    loss_name: str = "ce"
    loss_warmup_epochs: int = 0
    focal_gamma: float = 0.0
    class_balanced_beta: float = 0.999
    ldam_max_margin: float = 0.5
    ldam_scale: float = 30.0
    sampler_mode: str = "auto"
    sample_weight_usage: str = "loss"
    amp: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    jsd_weight: float = 0.0
    distillation_alpha: float = 0.0
    distillation_temperature: float = 1.0
    ema_decay: float = 0.0
    output_dir: Path = Path("outputs")


@dataclass
class InferConfig:
    batch_size: int = 64
    tta: bool = False
    amp: bool = False
    output_csv: Path = Path("submission.csv")


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def _coerce_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    annotations = getattr(instance, "__annotations__", {})
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key: {key}")
        expected = str(annotations.get(key, ""))
        if (
            expected.endswith("Path")
            or "Path" in expected
            or key.endswith(("_dir", "_csv", "_root"))
        ):
            value = _coerce_path(value)
        elif key in {"output_dir", "output_csv"}:
            value = Path(value)
        setattr(instance, key, value)
    return instance


def load_config(path: Path | None) -> AppConfig:
    config = AppConfig()
    if path is None:
        _validate_config(config)
        return config
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    for section_name, section_values in raw.items():
        section = getattr(config, section_name, None)
        if section is None:
            raise ValueError(f"Unknown config section: {section_name}")
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section must be a mapping: {section_name}")
        _update_dataclass(section, section_values)
    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    supported_losses = {"ce", "focal", "class_balanced", "class_balanced_focal", "balanced_softmax", "ldam"}
    if config.train.loss_name not in supported_losses:
        raise ValueError(
            "train.loss_name must be one of: ce, focal, class_balanced, class_balanced_focal, balanced_softmax, ldam"
        )
    if config.train.focal_gamma < 0:
        raise ValueError("train.focal_gamma must be non-negative")
    if config.train.loss_warmup_epochs < 0:
        raise ValueError("train.loss_warmup_epochs must be non-negative")
    if config.train.layer_decay <= 0 or config.train.layer_decay > 1:
        raise ValueError("train.layer_decay must be in (0, 1]")
    if not 0 <= config.train.class_balanced_beta < 1:
        raise ValueError("train.class_balanced_beta must be in [0, 1)")
    if config.train.ldam_max_margin <= 0:
        raise ValueError("train.ldam_max_margin must be positive")
    if config.train.ldam_scale <= 0:
        raise ValueError("train.ldam_scale must be positive")
    if config.train.sampler_mode not in {"auto", "none", "weighted", "sample_weighted"}:
        raise ValueError("train.sampler_mode must be one of: auto, none, weighted, sample_weighted")
    if config.train.sample_weight_usage not in {"loss", "sampler", "both"}:
        raise ValueError("train.sample_weight_usage must be one of: loss, sampler, both")
    if config.train.label_smoothing < 0 or config.train.label_smoothing >= 1:
        raise ValueError("train.label_smoothing must be in [0, 1)")
    if config.train.mixup_alpha < 0 or config.train.cutmix_alpha < 0:
        raise ValueError("train.mixup_alpha and train.cutmix_alpha must be non-negative")
    if config.train.jsd_weight < 0:
        raise ValueError("train.jsd_weight must be non-negative")
    if config.train.distillation_alpha < 0 or config.train.distillation_alpha > 1:
        raise ValueError("train.distillation_alpha must be in [0, 1]")
    if config.train.distillation_temperature <= 0:
        raise ValueError("train.distillation_temperature must be positive")
    if config.data.augment_policy == "augmix_jsd":
        if config.train.jsd_weight <= 0:
            raise ValueError("train.jsd_weight must be positive when data.augment_policy is augmix_jsd")
        if config.train.mixup_alpha > 0 or config.train.cutmix_alpha > 0:
            raise ValueError("augmix_jsd is intentionally mutually exclusive with MixUp/CutMix")
    if config.infer.batch_size <= 0:
        raise ValueError("infer.batch_size must be positive")


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
