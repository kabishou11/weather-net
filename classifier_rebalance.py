from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.weather_net.data import ManifestRow, build_manifest_from_csv, build_manifest_from_image_folder
from src.weather_net.datasets import WeatherImageDataset, build_transforms
from src.weather_net.models import create_classifier
from src.weather_net.training import train_one_epoch


HEAD_KEY_CANDIDATES = (
    "classifier.weight",
    "head.weight",
    "fc.weight",
    "head.fc.weight",
    "head.classifier.weight",
    "context_head.weight",
    "module.classifier.weight",
    "module.head.weight",
    "module.fc.weight",
    "module.head.fc.weight",
    "module.head.classifier.weight",
    "module.context_head.weight",
)


def _bias_key_for_head_key(head_key: str) -> str:
    if not head_key.endswith(".weight"):
        raise ValueError("classifier head key must end with .weight")
    return f"{head_key[:-len('.weight')]}.bias"


def _resolve_head_weight_key_from_prefix(state_dict: Mapping[str, torch.Tensor], head_prefix: str) -> str:
    if head_prefix == "auto":
        return _resolve_head_key(state_dict, "auto")
    if head_prefix.endswith(".weight"):
        return _resolve_head_key(state_dict, head_prefix)
    return _resolve_head_key(state_dict, f"{head_prefix}.weight")


def validate_class_to_idx(class_to_idx: Mapping[str, int]) -> dict[str, int]:
    mapping = {str(key): int(value) for key, value in class_to_idx.items()}
    values = sorted(mapping.values())
    if values != list(range(len(values))):
        raise ValueError("class_to_idx must use contiguous unique ids from 0")
    return mapping


def class_names_from_mapping(class_to_idx: Mapping[str, int]) -> list[str]:
    mapping = validate_class_to_idx(class_to_idx)
    return [name for name, _idx in sorted(mapping.items(), key=lambda item: item[1])]


def freeze_backbone_for_classifier_retraining(model: torch.nn.Module, head_prefix: str = "auto") -> list[str]:
    if head_prefix == "auto":
        state_keys = {name: parameter for name, parameter in model.named_parameters()}
        head_key = _resolve_head_key(state_keys, "auto")
        head_prefix = head_key.rsplit(".", 1)[0]
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        is_head = name == head_prefix or name.startswith(f"{head_prefix}.")
        parameter.requires_grad_(is_head)
        if is_head:
            trainable.append(name)
    if not trainable:
        raise ValueError(f"classifier head parameters not found for prefix: {head_prefix}")
    return trainable


def build_classifier_balanced_sampler(
    rows: list[ManifestRow],
    mode: str = "class_balanced",
) -> WeightedRandomSampler | None:
    if mode not in {"none", "class_balanced", "sqrt"}:
        raise ValueError("mode must be one of: none, class_balanced, sqrt")
    if mode == "none":
        return None
    labels: list[int] = []
    for row in rows:
        if row.label is None:
            raise ValueError("classifier re-training rows must be labeled")
        labels.append(int(row.label))
    if not labels:
        raise ValueError("classifier re-training requires at least one row")
    counts = np.bincount(labels).astype(float)
    exponent = 1.0 if mode == "class_balanced" else 0.5
    weights = [1.0 / (counts[label] ** exponent) for label in labels]
    if min(weights) == max(weights):
        return None
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _loader_kwargs(num_workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def load_training_manifest_from_args(
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None = None,
    class_to_idx: dict[str, int] | None = None,
) -> list[ManifestRow]:
    if train_csv is not None:
        rows, _mapping = build_manifest_from_csv(train_csv, image_root=image_root, class_to_idx=class_to_idx)
    elif train_dir is not None:
        rows, _mapping = build_manifest_from_image_folder(train_dir, class_to_idx=class_to_idx)
    else:
        raise ValueError("Set --train-csv or --train-dir for classifier re-training")
    non_labeled_sources = sorted({row.source for row in rows if row.source != "labeled"})
    if non_labeled_sources:
        raise ValueError(
            "classifier re-training should use only labeled rows; "
            f"found sources: {non_labeled_sources}"
        )
    return rows


def _resolve_user_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _row_id(row: ManifestRow) -> str:
    return row.image_id or str(row.path)


def _row_digest(rows: list[ManifestRow]) -> str:
    payload = [
        {
            "image": _row_id(row),
            "label": row.label_name,
            "source": row.source,
            "confidence": float(row.confidence),
            "sample_weight": float(row.sample_weight),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_fold(checkpoint: Mapping[str, Any]) -> int | None:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        data_config = config.get("data")
        if isinstance(data_config, dict):
            folds = int(data_config.get("folds", 1))
            if folds <= 1:
                return None
    if "fold" not in checkpoint:
        return None
    fold = checkpoint["fold"]
    if fold is None:
        return None
    try:
        return int(fold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint fold must be an integer: {fold!r}") from error


def _source_counts_are_labeled_only(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    normalized = {str(source): int(count) for source, count in value.items()}
    return set(normalized) == {"labeled"} and normalized["labeled"] > 0


def validate_rebalance_manifest_summary(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    train_csv: Path | None,
    summary_path: Path | None,
    rows: list[ManifestRow] | None = None,
) -> dict[str, object] | None:
    fold = _checkpoint_fold(checkpoint)
    if fold is None:
        return None
    if summary_path is None:
        raise ValueError(
            "rebalance manifest summary is required for fold checkpoint; "
            "pass --rebalance-manifest-summary from classifier_rebalance_grid.py --prepare-folds"
        )
    if train_csv is None:
        raise ValueError("fold-safe classifier re-training requires --train-csv from the rebalance manifest summary")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"rebalance manifest summary is not valid JSON: {summary_path}") from error
    if not isinstance(summary, dict):
        raise ValueError("rebalance manifest summary must be a JSON object")
    fold_entries = summary.get("folds")
    if not isinstance(fold_entries, list):
        raise ValueError("rebalance manifest summary is missing folds")
    fold_entry = None
    for item in fold_entries:
        if isinstance(item, dict) and int(item.get("fold", -1)) == fold:
            fold_entry = item
            break
    if fold_entry is None:
        raise ValueError(f"rebalance manifest summary has no entry for checkpoint fold {fold}")
    expected_train_csv = fold_entry.get("train_csv")
    if not isinstance(expected_train_csv, str) or not expected_train_csv:
        raise ValueError(f"rebalance manifest summary fold {fold} is missing train_csv")
    actual_train_csv = _resolve_user_path(train_csv)
    expected_train_path = _resolve_user_path(Path(expected_train_csv))
    if actual_train_csv != expected_train_path:
        raise ValueError(
            "train_csv does not match fold-safe train_csv for checkpoint fold "
            f"{fold}: {actual_train_csv} != {expected_train_path}"
        )
    if not _source_counts_are_labeled_only(fold_entry.get("train_source_counts")):
        raise ValueError(f"rebalance manifest summary fold {fold} train_source_counts must be labeled-only")
    if not _source_counts_are_labeled_only(fold_entry.get("val_source_counts")):
        raise ValueError(f"rebalance manifest summary fold {fold} val_source_counts must be labeled-only")
    if int(fold_entry.get("train_rows", 0)) <= 0 or int(fold_entry.get("val_rows", 0)) <= 0:
        raise ValueError(f"rebalance manifest summary fold {fold} must have positive train_rows and val_rows")
    if rows is not None:
        current_train_digest = _row_digest(rows)
        expected_train_digest = fold_entry.get("train_row_digest")
        if isinstance(expected_train_digest, str) and expected_train_digest and current_train_digest != expected_train_digest:
            raise ValueError(
                "train_csv row digest does not match fold-safe manifest summary for checkpoint fold "
                f"{fold}: {current_train_digest} != {expected_train_digest}"
            )
    val_csv = fold_entry.get("val_csv")
    if isinstance(val_csv, str) and val_csv:
        val_path = _resolve_user_path(Path(val_csv))
        if val_path == expected_train_path:
            raise ValueError(f"rebalance manifest summary fold {fold} uses the same train_csv and val_csv")
    return {
        "summary_json": str(_resolve_user_path(summary_path)),
        "checkpoint": str(_resolve_user_path(checkpoint_path)),
        "fold": fold,
        "train_csv": str(expected_train_path),
        "val_csv": val_csv,
        "train_rows": int(fold_entry["train_rows"]),
        "val_rows": int(fold_entry["val_rows"]),
        "train_source_counts": dict(fold_entry["train_source_counts"]),
        "val_source_counts": dict(fold_entry["val_source_counts"]),
        "labeled_row_digest": summary.get("labeled_row_digest"),
        "train_row_digest": fold_entry.get("train_row_digest"),
        "val_row_digest": fold_entry.get("val_row_digest"),
    }


def build_crt_loader(
    rows: list[ManifestRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
    sampler_mode: str,
    transform_backend: str = "auto",
) -> DataLoader:
    dataset = WeatherImageDataset(
        rows,
        transform=build_transforms(
            image_size=image_size,
            train=True,
            policy="standard",
            backend=transform_backend,
        ),
    )
    sampler = build_classifier_balanced_sampler(rows, mode=sampler_mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **_loader_kwargs(num_workers),
    )


def _resolve_head_key(state_dict: Mapping[str, torch.Tensor], head_key: str) -> str:
    if head_key != "auto":
        if head_key not in state_dict:
            raise ValueError(f"classifier head not found: {head_key}")
        return head_key
    matches = [key for key in HEAD_KEY_CANDIDATES if key in state_dict]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple classifier head candidates found: {matches}; pass --head-key")
    suffix_matches = [
        key
        for key, value in state_dict.items()
        if key.endswith((".classifier.weight", ".head.weight", ".fc.weight", ".head.fc.weight", ".head.classifier.weight"))
        and isinstance(value, torch.Tensor)
        and value.ndim == 2
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise ValueError(f"Multiple classifier head candidates found: {suffix_matches}; pass --head-key")
    raise ValueError("classifier head weight not found; pass --head-key")


def apply_tau_norm_to_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    tau: float,
    head_key: str = "auto",
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    resolved_key = _resolve_head_key(state_dict, head_key)
    weight = state_dict[resolved_key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("classifier head weight must be a 2D tensor")
    output = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}
    norms = weight.float().norm(dim=1, keepdim=True).clamp_min(eps)
    output[resolved_key] = (weight.float() / norms.pow(float(tau))).to(dtype=weight.dtype)
    return output


def fold_lws_into_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    log_scales: torch.Tensor,
    head_key: str = "auto",
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> dict[str, torch.Tensor]:
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("LWS scale range must be positive and ordered")
    resolved_key = _resolve_head_key(state_dict, head_key)
    weight = state_dict[resolved_key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("classifier head weight must be a 2D tensor")
    if log_scales.ndim != 1 or log_scales.numel() != weight.shape[0]:
        raise ValueError("log_scales must have one value per class")
    scales = log_scales.detach().to(device=weight.device, dtype=torch.float32).exp().view(-1, 1)
    if not torch.isfinite(scales).all():
        raise ValueError("LWS scales must be finite")
    if float(scales.min()) < min_scale or float(scales.max()) > max_scale:
        raise ValueError(f"LWS scales outside allowed range [{min_scale}, {max_scale}]")
    output = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}
    output[resolved_key] = (weight.float() * scales).to(dtype=weight.dtype)
    bias_key = _bias_key_for_head_key(resolved_key)
    bias = state_dict.get(bias_key)
    if isinstance(bias, torch.Tensor):
        output[bias_key] = (bias.float() * scales.flatten()).to(dtype=bias.dtype)
    if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in output.values()):
        raise ValueError("folded LWS checkpoint contains non-finite tensors")
    return output


def freeze_all_model_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _unpack_supervised_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError("supervised batch must contain images, targets, and sample weights")
    images, targets, sample_weights = batch[:3]
    return images, targets, sample_weights


def train_lws_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lws_log_scales: torch.nn.Parameter,
) -> float:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images, targets, sample_weights = _unpack_supervised_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base_logits = model(images)
        logits = base_logits * lws_log_scales.exp().view(1, -1)
        losses = criterion(logits, targets)
        loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(1, total_items)


def rebalance_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    tau: float = 1.0,
    head_key: str = "auto",
    map_location: str = "cpu",
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "model_state" not in checkpoint or not isinstance(checkpoint["model_state"], Mapping):
        raise ValueError("checkpoint must contain a model_state mapping")
    resolved_key = _resolve_head_key(checkpoint["model_state"], head_key)
    checkpoint = dict(checkpoint)
    checkpoint["model_state"] = apply_tau_norm_to_state_dict(
        checkpoint["model_state"],
        tau=tau,
        head_key=resolved_key,
    )
    checkpoint["rebalance"] = {
        "method": "tau_norm",
        "tau": float(tau),
        "head_key": resolved_key,
        "source_checkpoint": str(checkpoint_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "method": "tau_norm",
        "tau": float(tau),
        "head_key": resolved_key,
        "source": str(checkpoint_path),
        "output": str(output_path),
    }


def retrain_classifier_head(
    checkpoint_path: Path,
    output_path: Path,
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None = None,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    sampler_mode: str = "sqrt",
    head_prefix: str = "auto",
    device: str = "auto",
    num_workers: int = 2,
    transform_backend: str = "auto",
    amp: bool = False,
    rebalance_method: str = "crt",
    rebalance_manifest_summary: Path | None = None,
) -> dict[str, object]:
    if rebalance_method not in {"crt", "lws"}:
        raise ValueError("rebalance_method must be one of: crt, lws")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError("checkpoint must contain model_state and class_to_idx")
    class_to_idx = validate_class_to_idx(checkpoint["class_to_idx"])
    rebalance_manifest_audit = validate_rebalance_manifest_summary(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        train_csv=train_csv,
        summary_path=rebalance_manifest_summary,
    )
    rows = load_training_manifest_from_args(
        train_csv=train_csv,
        train_dir=train_dir,
        image_root=image_root,
        class_to_idx=class_to_idx,
    )
    if rebalance_manifest_audit is not None:
        rebalance_manifest_audit = validate_rebalance_manifest_summary(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            train_csv=train_csv,
            summary_path=rebalance_manifest_summary,
            rows=rows,
        )
    model_name = str(checkpoint.get("model_name", "convnext_tiny"))
    image_size = int(checkpoint.get("image_size", 224))
    model = create_classifier(
        model_name=model_name,
        num_classes=len(class_to_idx),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    if rebalance_method == "lws":
        freeze_all_model_parameters(model)
        trainable: list[str] = []
        state_keys = {name: parameter for name, parameter in model.named_parameters()}
        resolved_head_key = _resolve_head_weight_key_from_prefix(state_keys, head_prefix)
    else:
        trainable = freeze_backbone_for_classifier_retraining(model, head_prefix=head_prefix)
        state_keys = {name: parameter for name, parameter in model.named_parameters()}
        resolved_head_key = _resolve_head_weight_key_from_prefix(state_keys, head_prefix)
    loader = build_crt_loader(
        rows,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_mode=sampler_mode,
        transform_backend=transform_backend,
    )
    if rebalance_method == "lws":
        lws_log_scales = torch.nn.Parameter(torch.zeros(len(class_to_idx), device=device))
        optimizer = torch.optim.AdamW([lws_log_scales], lr=lr, weight_decay=weight_decay)
        scaler = None
    else:
        lws_log_scales = None
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        if rebalance_method == "lws":
            assert lws_log_scales is not None
            loss = train_lws_one_epoch(
                model,
                loader,
                optimizer,
                device=device,
                lws_log_scales=lws_log_scales,
            )
        else:
            assert scaler is not None
            loss = train_one_epoch(
                model,
                loader,
                optimizer,
                scaler,
                device=device,
                num_classes=len(class_to_idx),
                label_smoothing=0.0,
                mixup_alpha=0.0,
                cutmix_alpha=0.0,
                amp=amp,
                loss_name="ce",
                epoch=epoch,
            )
        losses.append(float(loss))
    output_checkpoint = dict(checkpoint)
    if rebalance_method == "lws":
        assert lws_log_scales is not None
        output_checkpoint["model_state"] = fold_lws_into_state_dict(
            model.state_dict(),
            log_scales=lws_log_scales.detach().cpu(),
            head_key=resolved_head_key,
        )
        learned_scales = [float(value) for value in lws_log_scales.detach().cpu().exp().tolist()]
        scale_by_class = dict(zip(class_names_from_mapping(class_to_idx), learned_scales))
    else:
        output_checkpoint["model_state"] = model.state_dict()
        learned_scales = None
        scale_by_class = None
    output_checkpoint["rebalance"] = {
        "method": rebalance_method,
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "sampler_mode": sampler_mode,
        "trainable": trainable,
        "head_key": resolved_head_key,
        "source_checkpoint": str(checkpoint_path),
    }
    if rebalance_manifest_audit is not None:
        output_checkpoint["rebalance"]["manifest_summary"] = rebalance_manifest_audit
    if learned_scales is not None:
        output_checkpoint["rebalance"]["scales"] = learned_scales
        output_checkpoint["rebalance"]["scale_by_class"] = scale_by_class
        output_checkpoint["rebalance"]["scales_folded_into"] = "weight_and_bias"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)
    return {
        "method": rebalance_method,
        "epochs": int(epochs),
        "lr": float(lr),
        "sampler_mode": sampler_mode,
        "trainable": trainable,
        "head_key": resolved_head_key,
        "scales": learned_scales,
        "scale_by_class": scale_by_class,
        "losses": losses,
        "source": str(checkpoint_path),
        "output": str(output_path),
        "manifest_summary": rebalance_manifest_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply classifier rebalancing to a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["tau_norm", "crt", "lws"], default="tau_norm")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--head-key", type=str, default="auto")
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sampler-mode", choices=["none", "class_balanced", "sqrt"], default="sqrt")
    parser.add_argument("--head-prefix", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--transform-backend", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rebalance-manifest-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "tau_norm":
        summary = rebalance_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            tau=args.tau,
            head_key=args.head_key,
            map_location=args.map_location,
        )
    else:
        summary = retrain_classifier_head(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            train_csv=args.train_csv,
            train_dir=args.train_dir,
            image_root=args.image_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            sampler_mode=args.sampler_mode,
            head_prefix=args.head_prefix,
            device=args.device,
            num_workers=args.num_workers,
            transform_backend=args.transform_backend,
            amp=args.amp,
            rebalance_method=args.method,
            rebalance_manifest_summary=args.rebalance_manifest_summary,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
