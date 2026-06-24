from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .data import (
    ManifestRow,
    build_unlabeled_manifest_from_csv,
    idx_to_class,
    list_images,
)
from .datasets import WeatherImageDataset, build_transforms
from .models import create_classifier
from .postprocess import DecisionParams, load_decision_params
from .submission import write_submission


def load_unlabeled_rows(
    test_dir: Path | None = None,
    test_csv: Path | None = None,
    image_root: Path | None = None,
    image_column: str | None = None,
) -> list[ManifestRow]:
    if test_csv is not None:
        return build_unlabeled_manifest_from_csv(
            test_csv,
            image_root=image_root,
            image_column=image_column,
        )
    if test_dir is not None:
        root = test_dir.expanduser().resolve()
        return [
            ManifestRow(path=path, image_id=path.relative_to(root).as_posix(), source="unlabeled")
            for path in list_images(root)
        ]
    raise ValueError("Set test_dir or test_csv")


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict[str, int], int]:
    checkpoint = torch.load(path, map_location=device)
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    model_name = checkpoint.get("model_name", "convnext_tiny")
    image_size = int(checkpoint.get("image_size", 224))
    model = create_classifier(
        model_name=model_name,
        num_classes=len(class_to_idx),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_to_idx, image_size


def normalize_checkpoint_weights(
    checkpoints: Sequence[Path],
    weights: Sequence[float] | None,
) -> list[float]:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if weights is None:
        return [1.0 / len(checkpoints)] * len(checkpoints)
    if len(weights) != len(checkpoints):
        raise ValueError("checkpoint weights must have the same count as checkpoints")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("checkpoint weights must have a positive sum")
    if any(weight < 0 for weight in weights):
        raise ValueError("checkpoint weights must be non-negative")
    return [float(weight) / total for weight in weights]


def resolve_checkpoint_weights(
    checkpoints: Sequence[Path],
    cli_weights: Sequence[float] | None,
    decision_params: DecisionParams | None,
) -> list[float]:
    if cli_weights is not None:
        return normalize_checkpoint_weights(checkpoints, cli_weights)
    if decision_params is None or decision_params.weights is None:
        return normalize_checkpoint_weights(checkpoints, None)
    if len(decision_params.weights) != len(checkpoints):
        raise ValueError("decision params weights must match checkpoint count")
    if decision_params.checkpoints is not None:
        checkpoint_names = [str(path) for path in checkpoints]
        if decision_params.checkpoints != checkpoint_names:
            raise ValueError(
                "decision params checkpoint order does not match --checkpoint; "
                "pass explicit --weights to override"
            )
    return normalize_checkpoint_weights(checkpoints, decision_params.weights)


def logit_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def combine_tta_logits(
    base_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    entropy_tolerance: float = 0.0,
) -> torch.Tensor:
    if base_logits.shape != augmented_logits.shape:
        raise ValueError("base_logits and augmented_logits must have the same shape")

    base_probs = torch.softmax(base_logits, dim=1)
    augmented_probs = torch.softmax(augmented_logits, dim=1)
    base_confidence, base_top1 = base_probs.max(dim=1)
    augmented_confidence, augmented_top1 = augmented_probs.max(dim=1)
    base_entropy = logit_entropy(base_logits)
    augmented_entropy = logit_entropy(augmented_logits)

    lower_confidence = augmented_confidence < base_confidence
    changed_top1_with_higher_entropy = (
        (augmented_top1 != base_top1)
        & (augmented_entropy > base_entropy + entropy_tolerance)
    )
    use_augmented = ~(lower_confidence | changed_top1_with_higher_entropy)
    averaged_logits = (base_logits + augmented_logits) / 2
    return torch.where(use_augmented.unsqueeze(1), averaged_logits, base_logits)


@torch.no_grad()
def predict_probabilities(
    checkpoints: Sequence[Path],
    rows: Sequence[ManifestRow],
    batch_size: int,
    device: str,
    tta: bool = False,
    num_workers: int = 2,
    checkpoint_weights: Sequence[float] | None = None,
    transform_backend: str = "auto",
    decision_params_path: Path | None = None,
) -> tuple[list[Path], list[list[float]], list[str], list[str], dict[str, object]]:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    decision_params = None

    models: list[torch.nn.Module] = []
    class_to_idx: dict[str, int] | None = None
    image_size = 224
    for checkpoint_path in checkpoints:
        model, mapping, image_size = load_checkpoint(checkpoint_path, device=device)
        if class_to_idx is None:
            class_to_idx = mapping
        elif mapping != class_to_idx:
            raise ValueError(f"Class mapping mismatch in {checkpoint_path}")
        models.append(model)

    assert class_to_idx is not None
    class_names = idx_to_class(class_to_idx)
    if decision_params_path is not None:
        decision_params = load_decision_params(
            decision_params_path,
            expected_class_names=class_names,
        )
    weights = resolve_checkpoint_weights(
        checkpoints=checkpoints,
        cli_weights=checkpoint_weights,
        decision_params=decision_params,
    )
    logit_bias = None
    temperature = 1.0
    if decision_params is not None:
        temperature = decision_params.temperature
        logit_bias = torch.tensor(decision_params.bias, dtype=torch.float32, device=device)
    dataset = WeatherImageDataset(
        rows,
        transform=build_transforms(
            image_size=image_size,
            train=False,
            policy="standard",
            backend=transform_backend,
        ),
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    probabilities: list[list[float]] = []
    image_paths: list[Path] = []
    image_ids: list[str] = []
    started = time.perf_counter()

    for images, _, paths, ids in loader:
        images = images.to(device)
        logits = torch.zeros((images.size(0), len(class_names)), device=device)
        for model, weight in zip(models, weights):
            logits = logits + (model(images) * weight)
        if tta:
            augmented_logits = torch.zeros((images.size(0), len(class_names)), device=device)
            flipped_images = torch.flip(images, dims=[3])
            for model, weight in zip(models, weights):
                augmented_logits = augmented_logits + (model(flipped_images) * weight)
            logits = combine_tta_logits(logits, augmented_logits)
        logits = logits / float(temperature)
        if logit_bias is not None:
            logits = logits + logit_bias
        probs = torch.softmax(logits, dim=1).cpu().tolist()
        probabilities.extend([[float(value) for value in row] for row in probs])
        image_paths.extend(Path(path) for path in paths)
        image_ids.extend(str(image_id) for image_id in ids)

    elapsed = time.perf_counter() - started
    stats = {
        "images": len(rows),
        "seconds": elapsed,
        "images_per_second": len(rows) / elapsed if elapsed > 0 else None,
        "checkpoints": [str(path) for path in checkpoints],
        "checkpoint_weights": weights,
        "tta": tta,
        "decision_params": str(decision_params_path) if decision_params_path is not None else None,
    }
    return image_paths, probabilities, class_names, image_ids, stats


@torch.no_grad()
def predict_logits(
    checkpoints: Sequence[Path],
    rows: Sequence[ManifestRow],
    batch_size: int,
    device: str,
    tta: bool = False,
    num_workers: int = 2,
    checkpoint_weights: Sequence[float] | None = None,
    transform_backend: str = "auto",
    decision_params_path: Path | None = None,
) -> tuple[list[Path], list[str], list[str], dict[str, object]]:
    image_paths, probabilities, class_names, image_ids, stats = predict_probabilities(
        checkpoints=checkpoints,
        rows=rows,
        batch_size=batch_size,
        device=device,
        tta=tta,
        num_workers=num_workers,
        checkpoint_weights=checkpoint_weights,
        transform_backend=transform_backend,
        decision_params_path=decision_params_path,
    )
    predictions = [class_names[max(range(len(row)), key=lambda idx: row[idx])] for row in probabilities]
    return image_paths, predictions, image_ids, stats


def run_inference(
    checkpoints: Sequence[Path],
    output_csv: Path,
    test_dir: Path | None,
    test_csv: Path | None,
    image_root: Path | None,
    batch_size: int,
    device: str,
    tta: bool,
    image_column: str | None = None,
    num_workers: int = 2,
    checkpoint_weights: Sequence[float] | None = None,
    transform_backend: str = "auto",
    sample_submission_path: Path | None = None,
    output_image_column: str = "image",
    output_label_column: str = "label",
    decision_params_path: Path | None = None,
) -> dict[str, object]:
    rows = load_unlabeled_rows(
        test_dir=test_dir,
        test_csv=test_csv,
        image_root=image_root,
        image_column=image_column,
    )
    image_paths, predictions, image_ids, stats = predict_logits(
        checkpoints=checkpoints,
        rows=rows,
        batch_size=batch_size,
        device=device,
        tta=tta,
        num_workers=num_workers,
        checkpoint_weights=checkpoint_weights,
        transform_backend=transform_backend,
        decision_params_path=decision_params_path,
    )
    write_submission(
        output_path=output_csv,
        image_paths=image_paths,
        predictions=predictions,
        image_ids=image_ids,
        image_column=output_image_column,
        label_column=output_label_column,
        sample_submission_path=sample_submission_path,
    )
    stats_path = output_csv.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
