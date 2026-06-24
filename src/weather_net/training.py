from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
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
    iter_kfold_splits,
    save_class_mapping,
    split_train_val,
)
from .datasets import WeatherImageDataset, build_transforms
from .metrics import ClassificationReport, classification_report
from .models import create_classifier
from .oof import OofArtifactPaths, OofRecord, write_oof_artifacts


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


def make_sampler(rows: Sequence[ManifestRow]) -> WeightedRandomSampler | None:
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
) -> tuple[DataLoader, DataLoader]:
    train_dataset = WeatherImageDataset(
        train_rows,
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
    sampler = make_sampler(train_rows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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


def _one_hot_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, num_classes=num_classes).float()


def _mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return images, _one_hot_targets(targets, num_classes)
    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[permutation]
    hard_targets = _one_hot_targets(targets, num_classes)
    mixed_targets = lam * hard_targets + (1 - lam) * hard_targets[permutation]
    return mixed_images, mixed_targets


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
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return images, _one_hot_targets(targets, num_classes)
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
    return mixed_images, mixed_targets


def apply_batch_augmentations(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    force_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if force_mode not in {None, "none", "mixup", "cutmix"}:
        raise ValueError("force_mode must be one of: none, mixup, cutmix")
    if force_mode == "none" or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return images, _one_hot_targets(targets, num_classes)
    if force_mode == "mixup":
        return _mixup_batch(images, targets, num_classes, mixup_alpha)
    if force_mode == "cutmix":
        return _cutmix_batch(images, targets, num_classes, cutmix_alpha)
    if mixup_alpha > 0 and cutmix_alpha > 0:
        if random.random() < 0.5:
            return _mixup_batch(images, targets, num_classes, mixup_alpha)
        return _cutmix_batch(images, targets, num_classes, cutmix_alpha)
    if mixup_alpha > 0:
        return _mixup_batch(images, targets, num_classes, mixup_alpha)
    return _cutmix_batch(images, targets, num_classes, cutmix_alpha)


def _soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    losses = -(soft_targets * log_probs).sum(dim=1)
    if sample_weights is not None:
        sample_weights = sample_weights.to(losses.device).float()
        return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return losses.mean()


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
    model_ema: ModelEma | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")
    use_amp = amp and device == "cuda"

    for images, targets, sample_weights in loader:
        images = images.to(device)
        targets = targets.to(device)
        sample_weights = sample_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            if mixup_alpha > 0 or cutmix_alpha > 0:
                images, soft_targets = apply_batch_augmentations(
                    images,
                    targets,
                    num_classes=num_classes,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                )
                logits = model(images)
                loss = _soft_cross_entropy(logits, soft_targets, sample_weights)
            else:
                logits = model(images)
                losses = criterion(logits, targets)
                if losses.ndim == 0:
                    loss = losses
                else:
                    loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(1, total_items)


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
    for images, targets, sample_weights in loader:
        images = images.to(device)
        targets = targets.to(device)
        sample_weights = sample_weights.to(device)
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
    if any(row.source == "pseudo" for row in rows):
        raise ValueError("Pseudo rows must not be used as OOF validation rows")
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
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    records: list[OofRecord] = []
    offset = 0
    for images, targets, _sample_weights in loader:
        images = images.to(device)
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
        )
        model = create_classifier(
            config.model.name,
            num_classes=len(class_names),
            pretrained=config.model.pretrained,
            drop_rate=config.model.dropout,
        ).to(device)
        model_ema = ModelEma(model, config.train.ema_decay) if config.train.ema_decay > 0 else None
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
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
                model_ema=model_ema,
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
        write_oof_artifacts(
            output_dir=output_dir / "oof",
            records=all_oof_records,
            class_names=class_names,
            class_to_idx=class_to_idx,
            metadata={
                "folds_requested": config.data.folds,
                "folds_actual": len(summary),
                "seed": config.data.seed,
                "model_name": config.model.name,
                "image_size": config.model.image_size,
                "transform_backend": config.data.transform_backend,
                "checkpoints": [str(path) for path in checkpoint_paths],
            },
        )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_paths
