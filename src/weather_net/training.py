from __future__ import annotations

import json
import hashlib
import math
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import AppConfig, resolve_device
from .data import (
    ManifestRow,
    build_manifest_from_csv,
    build_manifest_from_image_folder,
    idx_to_class,
    is_validation_source,
    iter_kfold_splits,
    save_class_mapping,
    split_train_val,
)
from .datasets import WeatherImageDataset, build_transforms
from .metrics import ClassificationReport, classification_report
from .models import create_classifier
from .oof import OofArtifactPaths, OofRecord, write_oof_artifacts


def _loader_kwargs(num_workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels(rows: Sequence[ManifestRow]) -> list[int]:
    labels: list[int] = []
    for row in rows:
        if row.label is None:
            raise ValueError("Training rows must be labeled")
        labels.append(row.label)
    return labels


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.module = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = model_state[key].detach().to(device=ema_value.device)
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


@dataclass(frozen=True)
class BatchAugmentationInfo:
    permutation: torch.Tensor | None
    lam: float = 1.0


def make_sampler(
    rows: Sequence[ManifestRow],
    sampler_mode: str = "auto",
    loss_name: str = "ce",
) -> WeightedRandomSampler | None:
    if sampler_mode not in {"auto", "none", "weighted", "sample_weighted"}:
        raise ValueError("sampler_mode must be one of: auto, none, weighted, sample_weighted")
    if sampler_mode == "none":
        return None
    if sampler_mode == "sample_weighted":
        weights = [float(row.sample_weight) for row in rows]
        if not weights:
            return None
        if not all(math.isfinite(weight) and weight > 0 for weight in weights):
            raise ValueError("sample_weighted sampler requires finite positive sample_weight values")
        if min(weights) == max(weights):
            return None
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    if sampler_mode == "auto" and ("class_balanced" in loss_name or loss_name in {"balanced_softmax", "ldam"}):
        return None
    labels = _labels(rows)
    counts = np.bincount(labels)
    if len(counts) == 0 or counts.min() == counts.max():
        return None
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def make_loaders(
    train_rows: Sequence[ManifestRow],
    val_rows: Sequence[ManifestRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment_policy: str = "standard",
    transform_backend: str = "auto",
    sampler_mode: str = "auto",
    loss_name: str = "ce",
    sample_weight_usage: str = "loss",
) -> tuple[DataLoader, DataLoader]:
    if sample_weight_usage not in {"loss", "sampler", "both"}:
        raise ValueError("sample_weight_usage must be one of: loss, sampler, both")
    sampler_rows = train_rows
    dataset_train_rows = train_rows
    if sample_weight_usage == "sampler":
        dataset_train_rows = [replace(row, sample_weight=1.0) for row in train_rows]
    elif sample_weight_usage == "loss" and sampler_mode == "sample_weighted":
        sampler_rows = [replace(row, sample_weight=1.0) for row in train_rows]
    train_dataset = WeatherImageDataset(
        dataset_train_rows,
        transform=build_transforms(
            image_size=image_size,
            train=True,
            policy=augment_policy,
            backend=transform_backend,
        ),
    )
    val_dataset = WeatherImageDataset(
        val_rows,
        transform=build_transforms(
            image_size=image_size,
            train=False,
            policy="standard",
            backend=transform_backend,
        ),
    )
    sampler = make_sampler(sampler_rows, sampler_mode=sampler_mode, loss_name=loss_name)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **_loader_kwargs(num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers),
    )
    return train_loader, val_loader


def load_training_manifest(config: AppConfig) -> tuple[list[ManifestRow], dict[str, int]]:
    if config.data.train_dir is not None:
        return build_manifest_from_image_folder(config.data.train_dir)
    if config.data.train_csv is not None:
        return build_manifest_from_csv(
            config.data.train_csv,
            image_root=config.data.image_root,
            image_column=config.data.image_column,
            label_column=config.data.label_column,
        )
    raise ValueError("Set data.train_dir or data.train_csv")


def _is_external_source(source: str) -> bool:
    return source == "external" or source.startswith("external_")


def _resolve_audit_output_csv(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("external merge audit is missing output_csv")
    return Path(value).expanduser().resolve()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_digest(row: ManifestRow) -> str:
    payload = {
        "image": row.image_id or str(row.path),
        "label": row.label_name or "",
        "source": row.source,
        "confidence": f"{row.confidence:.6f}",
        "sample_weight": f"{row.sample_weight:.6f}",
        "content_sha256": _hash_file(row.path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_external_merge_audit(rows: Sequence[ManifestRow], config: AppConfig) -> dict[str, object] | None:
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1
    if not any(_is_external_source(source) for source in source_counts):
        return None

    audit_path = config.data.external_audit_json
    if audit_path is None:
        raise ValueError(
            "external merge audit is required when training data contains external rows; "
            "set data.external_audit_json to the JSON produced by merge_external_training.py"
        )
    audit_path = audit_path.expanduser().resolve()
    if not audit_path.exists():
        raise ValueError(f"external merge audit does not exist: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"external merge audit is not valid JSON: {audit_path}") from error
    if not isinstance(audit, dict):
        raise ValueError("external merge audit must be a JSON object")

    if config.data.train_csv is not None:
        expected_csv = config.data.train_csv.expanduser().resolve()
        audited_csv = _resolve_audit_output_csv(audit.get("output_csv"))
        if audited_csv != expected_csv:
            raise ValueError(
                "external merge audit output_csv does not match data.train_csv: "
                f"{audited_csv} != {expected_csv}"
            )
    if int(audit.get("kept_external_rows", -1)) <= 0:
        raise ValueError("external merge audit reports no kept external rows")
    audit_source_counts = audit.get("source_counts")
    if not isinstance(audit_source_counts, dict):
        raise ValueError("external merge audit is missing source_counts")
    normalized_audit_source_counts = {str(source): int(count) for source, count in audit_source_counts.items()}
    if normalized_audit_source_counts != source_counts:
        raise ValueError(
            "external merge audit source_counts do not match training manifest: "
            f"{normalized_audit_source_counts} != {source_counts}"
        )
    audit_row_digests = audit.get("row_digest_sha256")
    if not isinstance(audit_row_digests, list) or not all(isinstance(value, str) for value in audit_row_digests):
        raise ValueError("external merge audit is missing row_digest_sha256")
    current_row_digests = [_row_digest(row) for row in rows]
    if audit_row_digests != current_row_digests:
        raise ValueError("external merge audit row_digest_sha256 does not match current training manifest")
    return {
        "path": str(audit_path),
        "kept_external_rows": int(audit["kept_external_rows"]),
        "source_counts": dict(sorted(normalized_audit_source_counts.items())),
        "row_digest_sha256": audit_row_digests,
        "effective_weight": audit.get("effective_weight"),
    }


def _decay_layer_id(name: str, head_layer_id: int | None = None) -> int:
    if name.startswith(("head.", "classifier.", "fc.")) or ".head." in name or ".classifier." in name or ".fc." in name:
        return 0 if head_layer_id is None else head_layer_id
    match = re.search(r"(?:^|\.)(?:stages|layers|blocks)\.(\d+)(?:\.|$)", name)
    if match is not None:
        return int(match.group(1)) + 1
    return 0


def _uses_weight_decay(name: str, parameter: nn.Parameter, no_weight_decay: bool) -> bool:
    if not no_weight_decay:
        return True
    if parameter.ndim <= 1:
        return False
    if name.endswith(".bias"):
        return False
    lowered = name.lower()
    return not any(token in lowered for token in ("norm", "bn", "ln", "bias"))


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    no_weight_decay: bool = False,
    layer_decay: float = 1.0,
) -> torch.optim.Optimizer:
    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if layer_decay <= 0 or layer_decay > 1:
        raise ValueError("layer_decay must be in (0, 1]")

    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("model has no trainable parameters")
    backbone_max_layer_id = max(_decay_layer_id(name) for name, _parameter in named_parameters)
    head_layer_id = backbone_max_layer_id + 1
    max_layer_id = max(_decay_layer_id(name, head_layer_id=head_layer_id) for name, _parameter in named_parameters)
    groups: dict[tuple[float, float], dict[str, object]] = {}
    for name, parameter in named_parameters:
        layer_id = _decay_layer_id(name, head_layer_id=head_layer_id)
        group_lr = float(lr) * (float(layer_decay) ** (max_layer_id - layer_id))
        group_weight_decay = float(weight_decay) if _uses_weight_decay(name, parameter, no_weight_decay) else 0.0
        key = (group_lr, group_weight_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": group_lr, "weight_decay": group_weight_decay}
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), lr=lr, weight_decay=weight_decay)


def _one_hot_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, num_classes=num_classes).float()


def unpack_training_batch(batch) -> tuple[torch.Tensor | tuple, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        images, targets, sample_weights = batch
        return images, targets, sample_weights, None
    if len(batch) == 4:
        images, targets, sample_weights, teacher_probs = batch
        if isinstance(teacher_probs, torch.Tensor) and teacher_probs.numel() == 0:
            teacher_probs = None
        return images, targets, sample_weights, teacher_probs
    raise ValueError("training batches must contain 3 or 4 items")


def _mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    if alpha <= 0:
        return images, _one_hot_targets(targets, num_classes), None, 1.0
    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[permutation]
    hard_targets = _one_hot_targets(targets, num_classes)
    mixed_targets = lam * hard_targets + (1 - lam) * hard_targets[permutation]
    return mixed_images, mixed_targets, permutation, float(lam)


def _rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    center_x = np.random.randint(width)
    center_y = np.random.randint(height)
    x1 = int(np.clip(center_x - cut_w // 2, 0, width))
    y1 = int(np.clip(center_y - cut_h // 2, 0, height))
    x2 = int(np.clip(center_x + cut_w // 2, 0, width))
    y2 = int(np.clip(center_y + cut_h // 2, 0, height))
    if x1 == x2:
        x2 = min(width, x1 + 1)
    if y1 == y2:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    if alpha <= 0:
        return images, _one_hot_targets(targets, num_classes), None, 1.0
    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(images.size(0), device=images.device)
    if images.size(0) > 1 and torch.equal(permutation, torch.arange(images.size(0), device=images.device)):
        permutation = torch.roll(permutation, shifts=1)
    _, _, height, width = images.shape
    x1, y1, x2, y2 = _rand_bbox(width=width, height=height, lam=lam)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    hard_targets = _one_hot_targets(targets, num_classes)
    mixed_targets = lam * hard_targets + (1 - lam) * hard_targets[permutation]
    return mixed_images, mixed_targets, permutation, float(lam)


def apply_batch_augmentations(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    sample_weights: torch.Tensor | None = None,
    force_mode: str | None = None,
    return_mix_info: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, BatchAugmentationInfo]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, BatchAugmentationInfo]
):
    return_weights = sample_weights is not None

    def _with_optional_weights(
        mixed_images: torch.Tensor,
        soft_targets: torch.Tensor,
        permutation: torch.Tensor | None = None,
        lam: float = 1.0,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, BatchAugmentationInfo]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, BatchAugmentationInfo]
    ):
        mix_info = BatchAugmentationInfo(permutation=permutation, lam=float(lam))
        if not return_weights:
            if return_mix_info:
                return mixed_images, soft_targets, mix_info
            return mixed_images, soft_targets
        assert sample_weights is not None
        if permutation is None:
            weighted_targets = soft_targets * sample_weights.unsqueeze(1)
            if return_mix_info:
                return mixed_images, soft_targets, weighted_targets, mix_info
            return mixed_images, soft_targets, weighted_targets
        hard_targets = _one_hot_targets(targets, num_classes)
        weighted_targets = (
            (float(lam) * hard_targets * sample_weights.unsqueeze(1))
            + ((1.0 - float(lam)) * hard_targets[permutation] * sample_weights[permutation].unsqueeze(1))
        )
        if return_mix_info:
            return mixed_images, soft_targets, weighted_targets, mix_info
        return mixed_images, soft_targets, weighted_targets

    if force_mode not in {None, "none", "mixup", "cutmix"}:
        raise ValueError("force_mode must be one of: none, mixup, cutmix")
    if force_mode == "none" or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return _with_optional_weights(images, _one_hot_targets(targets, num_classes))
    if force_mode == "mixup":
        mixed_images, soft_targets, permutation, lam = _mixup_batch(images, targets, num_classes, mixup_alpha)
        return _with_optional_weights(mixed_images, soft_targets, permutation, lam)
    if force_mode == "cutmix":
        mixed_images, soft_targets, permutation, lam = _cutmix_batch(images, targets, num_classes, cutmix_alpha)
        return _with_optional_weights(mixed_images, soft_targets, permutation, lam)
    if mixup_alpha > 0 and cutmix_alpha > 0:
        if random.random() < 0.5:
            mixed_images, soft_targets, permutation, lam = _mixup_batch(images, targets, num_classes, mixup_alpha)
            return _with_optional_weights(mixed_images, soft_targets, permutation, lam)
        mixed_images, soft_targets, permutation, lam = _cutmix_batch(images, targets, num_classes, cutmix_alpha)
        return _with_optional_weights(mixed_images, soft_targets, permutation, lam)
    if mixup_alpha > 0:
        mixed_images, soft_targets, permutation, lam = _mixup_batch(images, targets, num_classes, mixup_alpha)
        return _with_optional_weights(mixed_images, soft_targets, permutation, lam)
    mixed_images, soft_targets, permutation, lam = _cutmix_batch(images, targets, num_classes, cutmix_alpha)
    return _with_optional_weights(mixed_images, soft_targets, permutation, lam)


def _soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return weighted_soft_cross_entropy(logits, soft_targets, sample_weights=sample_weights)


def _apply_label_smoothing(
    soft_targets: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing <= 0:
        return soft_targets
    num_classes = soft_targets.shape[1]
    return soft_targets * (1.0 - label_smoothing) + (label_smoothing / num_classes)


def _apply_weighted_label_smoothing(
    weighted_targets: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing <= 0:
        return weighted_targets
    num_classes = weighted_targets.shape[1]
    row_weight = weighted_targets.sum(dim=1, keepdim=True)
    return weighted_targets * (1.0 - label_smoothing) + (row_weight * label_smoothing / num_classes)


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    loss_name: str,
    beta: float = 0.999,
) -> torch.Tensor | None:
    if loss_name not in {"ce", "focal", "class_balanced", "class_balanced_focal", "balanced_softmax", "ldam"}:
        raise ValueError("loss_name must be one of: ce, focal, class_balanced, class_balanced_focal, balanced_softmax, ldam")
    if "class_balanced" not in loss_name:
        return None
    if not 0 <= beta < 1:
        raise ValueError("class_balanced_beta must be in [0, 1)")
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    return weights / weights.mean().clamp_min(1e-12)


def compute_class_counts(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    return counts.clamp_min(1.0)


def compute_ldam_margins(
    labels: Sequence[int],
    num_classes: int,
    max_margin: float = 0.5,
) -> torch.Tensor:
    if max_margin <= 0:
        raise ValueError("ldam_max_margin must be positive")
    counts = compute_class_counts(labels, num_classes=num_classes)
    margins = 1.0 / torch.sqrt(torch.sqrt(counts))
    return margins * (float(max_margin) / margins.max().clamp_min(1e-12))


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be one of: mean, none")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    weights = soft_targets
    target_weight = soft_targets.sum(dim=1).to(logits.device, dtype=logits.dtype).clamp_min(1e-12)
    if class_weights is not None:
        class_weights = class_weights.to(logits.device, dtype=logits.dtype)
        weights = weights * class_weights
        target_weight = weights.sum(dim=1).clamp_min(1e-12)
    if focal_gamma > 0:
        probs = log_probs.exp()
        weights = weights * torch.pow(1.0 - probs, focal_gamma)
    losses = -(weights * log_probs).sum(dim=1)
    if reduction == "none":
        return losses
    normalizer = target_weight
    if sample_weights is not None:
        sample_weights = sample_weights.to(losses.device).float()
        normalizer = normalizer * sample_weights
        return (losses * sample_weights).sum() / normalizer.sum().clamp_min(1e-8)
    return losses.sum() / normalizer.sum().clamp_min(1e-8)


def distillation_kl_loss(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if alpha < 0 or alpha > 1:
        raise ValueError("distillation_alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("distillation_temperature must be positive")
    if teacher_probs.ndim != 2 or teacher_probs.shape != logits.shape:
        raise ValueError("teacher_probs must match logits shape")
    teacher_probs = teacher_probs.to(logits.device, dtype=logits.dtype)
    if not torch.isfinite(teacher_probs).all():
        raise ValueError("teacher_probs must be finite")
    if (teacher_probs < 0).any():
        raise ValueError("teacher_probs must be non-negative")
    row_sums = teacher_probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4):
        raise ValueError("teacher_probs rows must sum to 1")
    log_probs = torch.nn.functional.log_softmax(logits / float(temperature), dim=1)
    losses = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=1)
    losses = losses * (float(temperature) ** 2) * float(alpha)
    if sample_weights is not None:
        sample_weights = sample_weights.to(logits.device, dtype=logits.dtype)
        return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return losses.mean()


def mix_teacher_probabilities(
    teacher_probs: torch.Tensor,
    sample_weights: torch.Tensor,
    mix_info: BatchAugmentationInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mix_info.permutation is None:
        return teacher_probs, sample_weights
    lam = float(mix_info.lam)
    permutation = mix_info.permutation.to(teacher_probs.device)
    mixed_weights = (lam * sample_weights) + ((1.0 - lam) * sample_weights[permutation])
    weighted_teacher = (
        (lam * teacher_probs * sample_weights.unsqueeze(1))
        + ((1.0 - lam) * teacher_probs[permutation] * sample_weights[permutation].unsqueeze(1))
    )
    mixed_teacher = weighted_teacher / mixed_weights.unsqueeze(1).clamp_min(1e-8)
    return mixed_teacher, mixed_weights


def balanced_softmax_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_counts: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if class_counts.ndim != 1 or class_counts.numel() != logits.shape[1]:
        raise ValueError("class_counts must be a 1D tensor with one value per class")
    adjusted_logits = logits + class_counts.to(logits.device, dtype=logits.dtype).clamp_min(1.0).log().unsqueeze(0)
    return weighted_soft_cross_entropy(
        adjusted_logits,
        soft_targets,
        sample_weights=sample_weights,
        class_weights=None,
        focal_gamma=focal_gamma,
        reduction=reduction,
    )


def ldam_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    margins: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    margin_targets: torch.Tensor | None = None,
    scale: float = 30.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("ldam_scale must be positive")
    if margins.ndim != 1 or margins.numel() != logits.shape[1]:
        raise ValueError("margins must be a 1D tensor with one value per class")
    margins = margins.to(logits.device, dtype=logits.dtype)
    if margin_targets is None:
        margin_targets = soft_targets
    margin_targets = margin_targets.to(logits.device, dtype=logits.dtype)
    margin_per_sample = margin_targets * margins.unsqueeze(0)
    adjusted_logits = (logits - margin_per_sample) * float(scale)
    return weighted_soft_cross_entropy(
        adjusted_logits,
        soft_targets,
        sample_weights=sample_weights,
        class_weights=None,
        focal_gamma=0.0,
        reduction=reduction,
    )


def split_augmix_jsd_batch(images) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if isinstance(images, torch.Tensor):
        return images, []
    if not isinstance(images, (list, tuple)) or len(images) != 3:
        raise ValueError("augmix_jsd batches must contain exactly three image views")
    clean, aug1, aug2 = images
    if not isinstance(clean, torch.Tensor) or not isinstance(aug1, torch.Tensor) or not isinstance(aug2, torch.Tensor):
        raise ValueError("augmix_jsd image views must be tensors")
    if clean.shape != aug1.shape or clean.shape != aug2.shape:
        raise ValueError("augmix_jsd image views must have matching shapes")
    return clean, [aug1, aug2]


def augmix_jsd_loss(
    clean_logits: torch.Tensor,
    aug1_logits: torch.Tensor,
    aug2_logits: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if clean_logits.shape != aug1_logits.shape or clean_logits.shape != aug2_logits.shape:
        raise ValueError("AugMix JSD logits must have matching shapes")
    clean_logits = clean_logits.float()
    aug1_logits = aug1_logits.float()
    aug2_logits = aug2_logits.float()
    clean_probs = torch.softmax(clean_logits, dim=1)
    aug1_probs = torch.softmax(aug1_logits, dim=1)
    aug2_probs = torch.softmax(aug2_logits, dim=1)
    mixture_log_probs = ((clean_probs + aug1_probs + aug2_probs) / 3.0).clamp_min(1e-7).detach().log()
    losses = (
        torch.nn.functional.kl_div(mixture_log_probs, clean_probs, reduction="none").sum(dim=1)
        + torch.nn.functional.kl_div(mixture_log_probs, aug1_probs, reduction="none").sum(dim=1)
        + torch.nn.functional.kl_div(mixture_log_probs, aug2_probs, reduction="none").sum(dim=1)
    ) / 3.0
    if sample_weights is None:
        return losses.mean()
    sample_weights = sample_weights.to(losses.device, dtype=losses.dtype)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    amp: bool,
    loss_name: str = "ce",
    loss_warmup_epochs: int = 0,
    epoch: int = 1,
    focal_gamma: float = 0.0,
    class_weights: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    ldam_margins: torch.Tensor | None = None,
    ldam_scale: float = 30.0,
    model_ema: ModelEma | None = None,
    jsd_weight: float = 0.0,
    distillation_alpha: float = 0.0,
    distillation_temperature: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    use_amp = amp and device == "cuda"
    class_weights = None if class_weights is None else class_weights.to(device)
    class_counts = None if class_counts is None else class_counts.to(device)
    ldam_margins = None if ldam_margins is None else ldam_margins.to(device)
    effective_loss_name = effective_loss_name_for_epoch(
        loss_name=loss_name,
        epoch=epoch,
        loss_warmup_epochs=loss_warmup_epochs,
    )
    loss_warmup_active = effective_loss_name != loss_name
    effective_class_weights = None if loss_warmup_active else class_weights
    use_focal = effective_loss_name in {"focal", "class_balanced_focal"}
    use_balanced_softmax = effective_loss_name == "balanced_softmax"
    use_ldam = effective_loss_name == "ldam"

    if jsd_weight < 0:
        raise ValueError("jsd_weight must be non-negative")
    if distillation_alpha < 0 or distillation_alpha > 1:
        raise ValueError("distillation_alpha must be in [0, 1]")
    if distillation_temperature <= 0:
        raise ValueError("distillation_temperature must be positive")
    if use_balanced_softmax and class_counts is None:
        raise ValueError("class_counts is required when loss_name is balanced_softmax")
    if use_ldam and ldam_margins is None:
        raise ValueError("ldam_margins is required when loss_name is ldam")

    for batch in loader:
        images, targets, sample_weights, teacher_probs = unpack_training_batch(batch)
        images, augmix_views = split_augmix_jsd_batch(images)
        if jsd_weight > 0 and not augmix_views:
            raise ValueError("jsd_weight requires augmix_jsd batches with three image views")
        images = images.to(device, non_blocking=True)
        augmix_views = [view.to(device, non_blocking=True) for view in augmix_views]
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        if teacher_probs is not None:
            teacher_probs = teacher_probs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            distillation_sample_weights = sample_weights
            if augmix_views and (mixup_alpha > 0 or cutmix_alpha > 0):
                raise ValueError("augmix_jsd batches are intentionally mutually exclusive with MixUp/CutMix")
            if mixup_alpha > 0 or cutmix_alpha > 0:
                images, soft_targets, weighted_targets, mix_info = apply_batch_augmentations(
                    images,
                    targets,
                    num_classes=num_classes,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                    sample_weights=sample_weights,
                    return_mix_info=True,
                )
                if teacher_probs is not None:
                    teacher_probs, distillation_sample_weights = mix_teacher_probabilities(
                        teacher_probs,
                        sample_weights,
                        mix_info,
                    )
                soft_targets = _apply_label_smoothing(soft_targets, label_smoothing)
                weighted_targets = _apply_weighted_label_smoothing(weighted_targets, label_smoothing)
                logits = model(images)
                if use_balanced_softmax:
                    assert class_counts is not None
                    loss = balanced_softmax_cross_entropy(
                        logits,
                        weighted_targets,
                        class_counts=class_counts,
                        sample_weights=None,
                        focal_gamma=0.0,
                    )
                elif use_ldam:
                    assert ldam_margins is not None
                    loss = ldam_cross_entropy(
                        logits,
                        weighted_targets,
                        margins=ldam_margins,
                        sample_weights=None,
                        margin_targets=soft_targets,
                        scale=ldam_scale,
                    )
                else:
                    loss = weighted_soft_cross_entropy(
                        logits,
                        weighted_targets,
                        sample_weights=None,
                        class_weights=effective_class_weights,
                        focal_gamma=focal_gamma if use_focal else 0.0,
                    )
            else:
                logits = model(images)
                soft_targets = _one_hot_targets(targets, num_classes=num_classes)
                soft_targets = _apply_label_smoothing(soft_targets, label_smoothing)
                if use_balanced_softmax:
                    assert class_counts is not None
                    loss = balanced_softmax_cross_entropy(
                        logits,
                        soft_targets,
                        class_counts=class_counts,
                        sample_weights=sample_weights,
                        focal_gamma=0.0,
                    )
                elif use_ldam:
                    assert ldam_margins is not None
                    loss = ldam_cross_entropy(
                        logits,
                        soft_targets,
                        margins=ldam_margins,
                        sample_weights=sample_weights,
                        scale=ldam_scale,
                    )
                else:
                    loss = weighted_soft_cross_entropy(
                        logits,
                        soft_targets,
                        sample_weights=sample_weights,
                        class_weights=effective_class_weights,
                        focal_gamma=focal_gamma if use_focal else 0.0,
                    )
            if augmix_views:
                if jsd_weight <= 0:
                    raise ValueError("jsd_weight must be positive for augmix_jsd batches")
                aug1_logits = model(augmix_views[0])
                aug2_logits = model(augmix_views[1])
                loss = loss + (jsd_weight * augmix_jsd_loss(logits, aug1_logits, aug2_logits, sample_weights))
            if teacher_probs is not None and distillation_alpha > 0:
                loss = ((1.0 - float(distillation_alpha)) * loss) + distillation_kl_loss(
                    logits,
                    teacher_probs,
                    alpha=float(distillation_alpha),
                    temperature=distillation_temperature,
                    sample_weights=distillation_sample_weights,
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(1, total_items)


def effective_loss_name_for_epoch(loss_name: str, epoch: int, loss_warmup_epochs: int = 0) -> str:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if loss_warmup_epochs < 0:
        raise ValueError("loss_warmup_epochs must be non-negative")
    return "ce" if epoch <= loss_warmup_epochs else loss_name


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    class_names: Sequence[str],
) -> tuple[ClassificationReport, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss(reduction="none")
    for batch in loader:
        images, targets, sample_weights, _teacher_probs = unpack_training_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        logits = model(images)
        losses = criterion(logits, targets)
        loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        preds = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
    return classification_report(y_true, y_pred, class_names), total_loss / max(1, total_items)


@torch.no_grad()
def collect_oof_predictions(
    model: nn.Module,
    rows: Sequence[ManifestRow],
    class_names: Sequence[str],
    fold: int,
    checkpoint: str,
    model_name: str,
    device: str,
    image_size: int | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
    transform_backend: str = "auto",
    loader: DataLoader | None = None,
) -> list[OofRecord]:
    model.eval()
    if any(not is_validation_source(row) for row in rows):
        raise ValueError("OOF validation rows must come from labeled sources")
    if loader is None:
        if image_size is None:
            raise ValueError("image_size is required when loader is not provided")
        dataset = WeatherImageDataset(
            rows,
            transform=build_transforms(
                image_size=image_size,
                train=False,
                policy="standard",
                backend=transform_backend,
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(num_workers),
        )

    records: list[OofRecord] = []
    offset = 0
    for batch in loader:
        images, targets, _sample_weights, _teacher_probs = unpack_training_batch(batch)
        images = images.to(device, non_blocking=True)
        logits = model(images).detach().cpu().float()
        targets = targets.detach().cpu().tolist()
        batch_rows = rows[offset : offset + len(targets)]
        for row, target, logit_row in zip(batch_rows, targets, logits.tolist()):
            if row.label is None:
                raise ValueError("OOF rows must be labeled")
            true_idx = int(target)
            true_label = row.label_name or class_names[true_idx]
            records.append(
                OofRecord(
                    image_id=row.image_id or row.path.name,
                    image_path=str(row.path),
                    fold=fold,
                    source=row.source,
                    true_idx=true_idx,
                    true_label=true_label,
                    logits=[float(value) for value in logit_row],
                    checkpoint=checkpoint,
                    model_name=model_name,
                )
            )
        offset += len(targets)
    return records


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: dict[str, int],
    config: AppConfig,
    report: ClassificationReport,
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "model_name": config.model.name,
            "image_size": config.model.image_size,
            "fold": fold,
            "macro_f1": report.macro_f1,
            "config": {
                "model": config.model.__dict__,
                "train": {k: str(v) if isinstance(v, Path) else v for k, v in config.train.__dict__.items()},
                "data": {k: str(v) if isinstance(v, Path) else v for k, v in config.data.__dict__.items()},
            },
        },
        path,
    )


def build_fold_summary(
    fold: int,
    best_epoch: int,
    best_val_loss: float,
    checkpoint: str,
    report: ClassificationReport,
    oof_artifacts: OofArtifactPaths | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_macro_f1": report.macro_f1,
        "best_accuracy": report.accuracy,
        "best_val_loss": best_val_loss,
        "checkpoint": checkpoint,
        "per_class_f1": report.per_class_f1,
        "confusion_matrix": report.confusion_matrix,
    }
    if oof_artifacts is not None:
        summary.update(
            {
                "oof_predictions_csv": str(oof_artifacts.csv_path),
                "oof_probabilities_npz": str(oof_artifacts.npz_path),
                "oof_metrics_json": str(oof_artifacts.metrics_path),
            }
        )
    return summary


def _fold_plan(
    rows: Sequence[ManifestRow],
    folds: int,
    seed: int,
    val_fraction: float,
) -> Iterable[tuple[int, list[ManifestRow], list[ManifestRow]]]:
    if folds > 1:
        yield from iter_kfold_splits(rows, folds=folds, seed=seed)
        return
    train_rows, val_rows = split_train_val(rows, val_fraction=val_fraction, seed=seed)
    yield 0, train_rows, val_rows


def train_config(config: AppConfig, device_request: str = "auto") -> list[Path]:
    seed_everything(config.data.seed)
    device = resolve_device(device_request)
    rows, class_to_idx = load_training_manifest(config)
    external_merge_audit = validate_external_merge_audit(rows, config)
    class_names = idx_to_class(class_to_idx)
    output_dir = config.train.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_class_mapping(class_to_idx, output_dir / "class_to_idx.json")

    checkpoint_paths: list[Path] = []
    summary: list[dict[str, object]] = []
    all_oof_records: list[OofRecord] = []
    for fold, train_rows, val_rows in _fold_plan(
        rows,
        folds=config.data.folds,
        seed=config.data.seed,
        val_fraction=config.data.val_fraction,
    ):
        train_loader, val_loader = make_loaders(
            train_rows,
            val_rows,
            image_size=config.model.image_size,
            batch_size=config.train.batch_size,
            num_workers=config.data.num_workers,
            augment_policy=config.data.augment_policy,
            transform_backend=config.data.transform_backend,
            sampler_mode=config.train.sampler_mode,
            loss_name=config.train.loss_name,
            sample_weight_usage=config.train.sample_weight_usage,
        )
        model = create_classifier(
            config.model.name,
            num_classes=len(class_names),
            pretrained=config.model.pretrained,
            drop_rate=config.model.dropout,
        ).to(device)
        model_ema = ModelEma(model, config.train.ema_decay) if config.train.ema_decay > 0 else None
        class_weights = compute_class_weights(
            labels=_labels(train_rows),
            num_classes=len(class_names),
            loss_name=config.train.loss_name,
            beta=config.train.class_balanced_beta,
        )
        class_counts = (
            compute_class_counts(_labels(train_rows), num_classes=len(class_names))
            if config.train.loss_name == "balanced_softmax"
            else None
        )
        ldam_margins = (
            compute_ldam_margins(
                _labels(train_rows),
                num_classes=len(class_names),
                max_margin=config.train.ldam_max_margin,
            )
            if config.train.loss_name == "ldam"
            else None
        )
        optimizer = build_optimizer(
            model,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            no_weight_decay=config.train.no_weight_decay,
            layer_decay=config.train.layer_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.train.epochs),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device == "cuda")
        best_f1 = -math.inf
        best_epoch = 0
        best_val_loss = math.inf
        best_report: ClassificationReport | None = None
        best_path = output_dir / f"{config.model.name}_fold{fold}.pt"

        for epoch in range(1, config.train.epochs + 1):
            started = time.perf_counter()
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device=device,
                num_classes=len(class_names),
                label_smoothing=config.train.label_smoothing,
                mixup_alpha=config.train.mixup_alpha,
                cutmix_alpha=config.train.cutmix_alpha,
                amp=config.train.amp,
                loss_name=config.train.loss_name,
                loss_warmup_epochs=config.train.loss_warmup_epochs,
                epoch=epoch,
                focal_gamma=config.train.focal_gamma,
                class_weights=class_weights,
                class_counts=class_counts,
                ldam_margins=ldam_margins,
                ldam_scale=config.train.ldam_scale,
                model_ema=model_ema,
                jsd_weight=config.train.jsd_weight,
                distillation_alpha=config.train.distillation_alpha,
                distillation_temperature=config.train.distillation_temperature,
            )
            eval_model = model_ema.module if model_ema is not None else model
            report, val_loss = evaluate(eval_model, val_loader, device=device, class_names=class_names)
            scheduler.step()
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "loss_name": config.train.loss_name,
                        "effective_loss_name": effective_loss_name_for_epoch(
                            config.train.loss_name,
                            epoch=epoch,
                            loss_warmup_epochs=config.train.loss_warmup_epochs,
                        ),
                        "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "macro_f1": round(report.macro_f1, 6),
                        "accuracy": round(report.accuracy, 6),
                        "ema": model_ema is not None,
                        "seconds": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                )
            )
            if report.macro_f1 > best_f1:
                best_f1 = report.macro_f1
                best_epoch = epoch
                best_val_loss = val_loss
                best_report = report
                save_checkpoint(best_path, eval_model, class_to_idx, config, report, fold)

        checkpoint_paths.append(best_path)
        if best_report is None:
            raise RuntimeError(f"No checkpoint was saved for fold {fold}")
        best_model = create_classifier(
            config.model.name,
            num_classes=len(class_names),
            pretrained=False,
            drop_rate=config.model.dropout,
        ).to(device)
        best_checkpoint = torch.load(best_path, map_location=device)
        best_model.load_state_dict(best_checkpoint["model_state"])
        fold_oof_records = collect_oof_predictions(
            model=best_model,
            rows=val_rows,
            class_names=class_names,
            fold=fold,
            checkpoint=str(best_path),
            model_name=config.model.name,
            device=device,
            image_size=config.model.image_size,
            batch_size=config.train.batch_size,
            num_workers=config.data.num_workers,
            transform_backend=config.data.transform_backend,
        )
        all_oof_records.extend(fold_oof_records)
        fold_oof_artifacts = write_oof_artifacts(
            output_dir=output_dir / "oof" / f"fold{fold}",
            records=fold_oof_records,
            class_names=class_names,
            class_to_idx=class_to_idx,
            metadata={
                "fold": fold,
                "checkpoint": str(best_path),
                "model_name": config.model.name,
                "image_size": config.model.image_size,
                "transform_backend": config.data.transform_backend,
                "seed": config.data.seed,
            },
        )
        summary.append(
            build_fold_summary(
                fold=fold,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                checkpoint=str(best_path),
                report=best_report,
                oof_artifacts=fold_oof_artifacts,
            )
        )

    if all_oof_records:
        metadata: dict[str, object] = {
            "folds_requested": config.data.folds,
            "folds_actual": len(summary),
            "seed": config.data.seed,
            "model_name": config.model.name,
            "image_size": config.model.image_size,
            "transform_backend": config.data.transform_backend,
            "checkpoints": [str(path) for path in checkpoint_paths],
        }
        if external_merge_audit is not None:
            metadata["external_merge_audit"] = external_merge_audit
        write_oof_artifacts(
            output_dir=output_dir / "oof",
            records=all_oof_records,
            class_names=class_names,
            class_to_idx=class_to_idx,
            metadata=metadata,
        )

    if external_merge_audit is not None:
        for item in summary:
            item["external_merge_audit"] = external_merge_audit
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_paths
