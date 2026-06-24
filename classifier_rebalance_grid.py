from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from classifier_rebalance import rebalance_checkpoint, retrain_classifier_head
from src.weather_net.data import (
    ManifestRow,
    build_manifest_from_csv,
    build_manifest_from_image_folder,
    idx_to_class,
    is_validation_source,
    iter_kfold_splits,
    load_class_mapping,
)
from validate import validate_checkpoint


@dataclass
class RebalanceCandidate:
    name: str
    method: str
    checkpoint: Path
    output: Path
    tau: float | None = None
    sampler_mode: str | None = None
    metrics_json: Path | None = None


@dataclass(frozen=True)
class FoldSafeManifest:
    fold: int
    train_csv: Path
    val_csv: Path
    train_rows: int
    val_rows: int
    train_source_counts: dict[str, int]
    val_source_counts: dict[str, int]
    train_row_digest: str
    val_row_digest: str


def _tau_name(value: float) -> str:
    return f"tau{value:.2f}".replace(".", "p")


def _validate_unique_candidates(candidates: Sequence[RebalanceCandidate]) -> None:
    names = [candidate.name for candidate in candidates]
    outputs = [candidate.output for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("duplicate rebalance candidate names generated")
    if len(set(outputs)) != len(outputs):
        raise ValueError("duplicate rebalance candidate outputs generated")


def _source_counts(rows: Sequence[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    return dict(sorted(counts.items()))


def _row_id(row: ManifestRow) -> str:
    return row.image_id or str(row.path)


def _row_digest(rows: Sequence[ManifestRow]) -> str:
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


def _write_manifest_rows_csv(path: Path, rows: Sequence[ManifestRow], class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_teacher = any(row.teacher_probs is not None for row in rows)
    fieldnames = ["image", "label", "source", "confidence", "sample_weight"]
    if include_teacher:
        fieldnames.extend(f"teacher_{class_name}" for class_name in class_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in rows:
            if row.label_name is None:
                raise ValueError(f"manifest row is missing label_name: {row.path}")
            image = row.image_id or str(row.path)
            values = [
                image,
                row.label_name,
                row.source,
                f"{float(row.confidence):.6f}",
                f"{float(row.sample_weight):.6f}",
            ]
            if include_teacher:
                if row.teacher_probs is None or len(row.teacher_probs) != len(class_names):
                    raise ValueError("all rows must have teacher probabilities when teacher columns are written")
                values.extend(f"{float(value):.8f}" for value in row.teacher_probs)
            writer.writerow(values)


def load_rebalance_manifest(
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None = None,
    class_map: Path | None = None,
) -> tuple[list[ManifestRow], dict[str, int]]:
    class_to_idx = load_class_mapping(class_map) if class_map is not None else None
    if train_csv is not None and train_dir is not None:
        raise ValueError("Use only one of --train-csv or --train-dir")
    if train_csv is not None:
        if class_to_idx is None:
            raise ValueError("fold-safe rebalance requires --class-map for CSV input")
        return build_manifest_from_csv(train_csv, image_root=image_root, class_to_idx=class_to_idx)
    if train_dir is not None:
        return build_manifest_from_image_folder(train_dir, class_to_idx=class_to_idx)
    raise ValueError("Set --train-csv or --train-dir")


def build_fold_safe_rebalance_manifests(
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None,
    class_map: Path | None,
    output_dir: Path,
    folds: int,
    seed: int,
) -> dict[str, object]:
    if folds < 2:
        raise ValueError("fold-safe rebalance requires at least 2 folds")
    rows, class_to_idx = load_rebalance_manifest(
        train_csv=train_csv,
        train_dir=train_dir,
        image_root=image_root,
        class_map=class_map,
    )
    excluded_rows = [row for row in rows if not is_validation_source(row)]
    labeled_rows = [row for row in rows if is_validation_source(row)]
    if not labeled_rows:
        raise ValueError("fold-safe rebalance requires labeled rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = idx_to_class(class_to_idx)
    manifests: list[FoldSafeManifest] = []
    for fold, train_rows, val_rows in iter_kfold_splits(labeled_rows, folds=folds, seed=seed):
        leaked_val_sources = sorted({row.source for row in val_rows if not is_validation_source(row)})
        if leaked_val_sources:
            raise ValueError(f"validation fold contains non-labeled sources: {leaked_val_sources}")
        leaked_train_sources = sorted({row.source for row in train_rows if not is_validation_source(row)})
        if leaked_train_sources:
            raise ValueError(f"rebalance train fold contains non-labeled sources: {leaked_train_sources}")
        val_ids = {_row_id(row) for row in val_rows}
        leaked_train_ids = [
            _row_id(row)
            for row in train_rows
            if _row_id(row) in val_ids
        ]
        if leaked_train_ids:
            raise ValueError(f"fold train rows overlap validation rows: {leaked_train_ids[:5]}")
        val_paths = {str(row.path.resolve()) for row in val_rows}
        leaked_train_paths = [str(row.path.resolve()) for row in train_rows if str(row.path.resolve()) in val_paths]
        if leaked_train_paths:
            raise ValueError(f"fold train paths overlap validation paths: {leaked_train_paths[:5]}")
        train_path = output_dir / f"fold{fold}_rebalance_train.csv"
        val_path = output_dir / f"fold{fold}_rebalance_val.csv"
        _write_manifest_rows_csv(train_path, train_rows, class_names=class_names)
        _write_manifest_rows_csv(val_path, val_rows, class_names=class_names)
        manifests.append(
            FoldSafeManifest(
                fold=fold,
                train_csv=train_path,
                val_csv=val_path,
                train_rows=len(train_rows),
                val_rows=len(val_rows),
                train_source_counts=_source_counts(train_rows),
                val_source_counts=_source_counts(val_rows),
                train_row_digest=_row_digest(train_rows),
                val_row_digest=_row_digest(val_rows),
            )
        )
    summary = {
        "folds": [
            {
                "fold": item.fold,
                "train_csv": str(item.train_csv.resolve()),
                "val_csv": str(item.val_csv.resolve()),
                "train_rows": item.train_rows,
                "val_rows": item.val_rows,
                "train_source_counts": item.train_source_counts,
                "val_source_counts": item.val_source_counts,
                "train_row_digest": item.train_row_digest,
                "val_row_digest": item.val_row_digest,
            }
            for item in manifests
        ],
        "class_names": class_names,
        "source_counts": _source_counts(rows),
        "image_root": str(image_root) if image_root is not None else (str(train_dir) if train_dir is not None else None),
        "excluded_source_counts": _source_counts(excluded_rows),
        "excluded_rows": len(excluded_rows),
        "labeled_rows": len(labeled_rows),
        "labeled_row_digest": _row_digest(labeled_rows),
        "fold_count": len(manifests),
    }
    summary_path = output_dir / "fold_safe_rebalance_manifests.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**summary, "summary_json": str(summary_path)}


def build_rebalance_grid(
    checkpoint: Path,
    output_dir: Path,
    tau_values: Sequence[float],
    crt_sampler_modes: Sequence[str],
    lws_sampler_modes: Sequence[str] = (),
) -> list[RebalanceCandidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    candidates: list[RebalanceCandidate] = []
    for tau in tau_values:
        value = float(tau)
        if value < 0:
            raise ValueError("tau values must be non-negative")
        name = _tau_name(value)
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="tau_norm",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                tau=value,
            )
        )
    valid_sampler_modes = {"none", "sqrt", "class_balanced"}
    for sampler_mode in crt_sampler_modes:
        if sampler_mode not in valid_sampler_modes:
            raise ValueError("crt sampler modes must be one of: none, sqrt, class_balanced")
        name = f"crt_{sampler_mode}"
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="crt",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                sampler_mode=sampler_mode,
            )
        )
    for sampler_mode in lws_sampler_modes:
        if sampler_mode not in valid_sampler_modes:
            raise ValueError("lws sampler modes must be one of: none, sqrt, class_balanced")
        name = f"lws_{sampler_mode}"
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="lws",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                sampler_mode=sampler_mode,
            )
        )
    _validate_unique_candidates(candidates)
    return candidates


def _read_metrics(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "macro_f1" not in data:
        raise ValueError(f"metrics JSON missing macro_f1: {path}")
    per_class = data.get("per_class_f1", {})
    if not isinstance(per_class, dict):
        raise ValueError(f"metrics JSON per_class_f1 must be a mapping: {path}")
    return data


def summarize_rebalance_grid(
    candidates: Sequence[RebalanceCandidate],
    baseline_name: str = "baseline",
    min_delta_macro_f1: float = 0.0,
    min_per_class_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not 0 <= min_per_class_f1 <= 1:
        raise ValueError("min_per_class_f1 must be in [0, 1]")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")
    results: list[dict[str, object]] = []
    baseline_macro_f1: float | None = None
    for candidate in candidates:
        if candidate.metrics_json is None:
            raise ValueError(f"candidate metrics_json is required: {candidate.name}")
        metrics = _read_metrics(candidate.metrics_json)
        per_class = {str(label): float(score) for label, score in metrics.get("per_class_f1", {}).items()}
        macro_f1 = float(metrics["macro_f1"])
        min_class_f1 = min(per_class.values()) if per_class else 0.0
        item = {
            "name": candidate.name,
            "method": candidate.method,
            "checkpoint": str(candidate.output),
            "metrics_json": str(candidate.metrics_json),
            "macro_f1": macro_f1,
            "per_class_f1": per_class,
            "min_per_class_f1": float(min_class_f1),
            "tau": candidate.tau,
            "sampler_mode": candidate.sampler_mode,
        }
        results.append(item)
        if candidate.name == baseline_name:
            baseline_macro_f1 = macro_f1
    if baseline_macro_f1 is None:
        raise ValueError(f"Missing baseline candidate: {baseline_name}")
    for item in results:
        delta = float(item["macro_f1"]) - baseline_macro_f1
        item["delta_macro_f1_vs_baseline"] = float(delta)
        if float(item["min_per_class_f1"]) < min_per_class_f1:
            item["selection_status"] = "rejected_min_per_class_f1"
        elif item["name"] != baseline_name and delta < min_delta_macro_f1:
            item["selection_status"] = "rejected_min_delta_macro_f1"
        else:
            item["selection_status"] = "candidate"
    candidates_kept = [item for item in results if item["selection_status"] == "candidate"]
    if not candidates_kept:
        raise ValueError("No candidates passed rebalance grid gates")
    top_score = max(float(item["macro_f1"]) for item in candidates_kept)
    near_top = [item for item in candidates_kept if top_score - float(item["macro_f1"]) <= tie_epsilon]
    best = sorted(
        near_top,
        key=lambda item: (
            0 if item["name"] == baseline_name else 1,
            -float(item["min_per_class_f1"]),
            str(item["name"]),
        ),
    )[0]
    return {
        "best_name": best["name"],
        "best_checkpoint": best["checkpoint"],
        "best_macro_f1": best["macro_f1"],
        "baseline_macro_f1": baseline_macro_f1,
        "gates": {
            "min_delta_macro_f1": min_delta_macro_f1,
            "min_per_class_f1": min_per_class_f1,
            "tie_epsilon": tie_epsilon,
        },
        "results": results,
    }


def write_summary(output_dir: Path, summary: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "classifier_rebalance_grid_summary.json"
    csv_path = output_dir / "classifier_rebalance_grid_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "method", "macro_f1", "delta_macro_f1_vs_baseline", "min_per_class_f1", "status", "checkpoint"])
        for item in summary["results"]:
            writer.writerow(
                [
                    item["name"],
                    item["method"],
                    f"{float(item['macro_f1']):.6f}",
                    f"{float(item['delta_macro_f1_vs_baseline']):.6f}",
                    f"{float(item['min_per_class_f1']):.6f}",
                    item["selection_status"],
                    item["checkpoint"],
                ]
            )
    return json_path, csv_path


def summarize_fold_rebalance_grid(
    candidates_by_fold: dict[int, Sequence[RebalanceCandidate]],
    baseline_name: str = "baseline",
    min_delta_macro_f1: float = 0.0,
    min_per_class_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not candidates_by_fold:
        raise ValueError("fold rebalance summary requires at least one fold")
    if not 0 <= min_per_class_f1 <= 1:
        raise ValueError("min_per_class_f1 must be in [0, 1]")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")

    fold_ids = sorted(candidates_by_fold)
    per_name: dict[str, list[dict[str, object]]] = {}
    baseline_macros: list[float] = []
    for fold in fold_ids:
        names_in_fold: set[str] = set()
        for candidate in candidates_by_fold[fold]:
            if candidate.metrics_json is None:
                raise ValueError(f"candidate metrics_json is required: fold {fold} {candidate.name}")
            metrics = _read_metrics(candidate.metrics_json)
            per_class = {str(label): float(score) for label, score in metrics.get("per_class_f1", {}).items()}
            macro_f1 = float(metrics["macro_f1"])
            item = {
                "fold": fold,
                "name": candidate.name,
                "method": candidate.method,
                "checkpoint": str(candidate.output),
                "metrics_json": str(candidate.metrics_json),
                "macro_f1": macro_f1,
                "per_class_f1": per_class,
                "min_per_class_f1": min(per_class.values()) if per_class else 0.0,
                "tau": candidate.tau,
                "sampler_mode": candidate.sampler_mode,
            }
            per_name.setdefault(candidate.name, []).append(item)
            names_in_fold.add(candidate.name)
            if candidate.name == baseline_name:
                baseline_macros.append(macro_f1)
        if baseline_name not in names_in_fold:
            raise ValueError(f"Missing baseline candidate for fold {fold}: {baseline_name}")
    if len(baseline_macros) != len(fold_ids):
        raise ValueError("baseline candidate count does not match fold count")
    incomplete = {name: len(items) for name, items in per_name.items() if len(items) != len(fold_ids)}
    if incomplete:
        raise ValueError(f"candidate names are not present in every fold: {incomplete}")

    baseline_mean = sum(baseline_macros) / len(baseline_macros)
    results: list[dict[str, object]] = []
    for name, fold_items in sorted(per_name.items()):
        macros = [float(item["macro_f1"]) for item in fold_items]
        min_class_scores = [float(item["min_per_class_f1"]) for item in fold_items]
        mean_macro = sum(macros) / len(macros)
        min_per_class_across_folds = min(min_class_scores) if min_class_scores else 0.0
        method = str(fold_items[0]["method"])
        delta = mean_macro - baseline_mean
        item = {
            "name": name,
            "method": method,
            "folds": fold_items,
            "mean_macro_f1": mean_macro,
            "baseline_mean_macro_f1": baseline_mean,
            "delta_mean_macro_f1_vs_baseline": delta,
            "min_per_class_f1_across_folds": min_per_class_across_folds,
            "checkpoint_by_fold": {str(item["fold"]): item["checkpoint"] for item in fold_items},
        }
        if min_per_class_across_folds < min_per_class_f1:
            item["selection_status"] = "rejected_min_per_class_f1"
        elif name != baseline_name and delta < min_delta_macro_f1:
            item["selection_status"] = "rejected_min_delta_macro_f1"
        else:
            item["selection_status"] = "candidate"
        results.append(item)
    kept = [item for item in results if item["selection_status"] == "candidate"]
    if not kept:
        raise ValueError("No fold candidates passed rebalance grid gates")
    top_score = max(float(item["mean_macro_f1"]) for item in kept)
    near_top = [item for item in kept if top_score - float(item["mean_macro_f1"]) <= tie_epsilon]
    best = sorted(
        near_top,
        key=lambda item: (
            0 if item["name"] == baseline_name else 1,
            -float(item["min_per_class_f1_across_folds"]),
            str(item["name"]),
        ),
    )[0]
    return {
        "best_name": best["name"],
        "best_checkpoint_by_fold": best["checkpoint_by_fold"],
        "best_mean_macro_f1": best["mean_macro_f1"],
        "baseline_mean_macro_f1": baseline_mean,
        "fold_count": len(fold_ids),
        "folds": fold_ids,
        "gates": {
            "min_delta_macro_f1": min_delta_macro_f1,
            "min_per_class_f1": min_per_class_f1,
            "tie_epsilon": tie_epsilon,
        },
        "results": results,
    }


def _load_fold_manifest_summary(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rebalance manifest summary must be a JSON object")
    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ValueError("rebalance manifest summary is missing folds")
    by_fold: dict[int, dict[str, object]] = {}
    for item in folds:
        if not isinstance(item, dict):
            raise ValueError("rebalance manifest fold entries must be objects")
        fold = int(item.get("fold", -1))
        if fold < 0:
            raise ValueError(f"invalid fold entry: {item}")
        if fold in by_fold:
            raise ValueError(f"duplicate fold entry in rebalance manifest summary: {fold}")
        train_csv = item.get("train_csv")
        val_csv = item.get("val_csv")
        if not isinstance(train_csv, str) or not train_csv:
            raise ValueError(f"rebalance manifest fold {fold} is missing train_csv")
        if not isinstance(val_csv, str) or not val_csv:
            raise ValueError(f"rebalance manifest fold {fold} is missing val_csv")
        by_fold[fold] = item
    return by_fold


def _checkpoint_fold_from_path(path: Path, map_location: str = "cpu") -> int:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "fold" not in checkpoint:
        raise ValueError(f"checkpoint is missing fold metadata: {path}")
    try:
        return int(checkpoint["fold"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint fold metadata must be an integer: {path}") from error


def run_all_fold_rebalance_grid(
    checkpoints: Sequence[Path],
    rebalance_manifest_summary: Path,
    output_dir: Path,
    tau_values: Sequence[float],
    crt_sampler_modes: Sequence[str],
    lws_sampler_modes: Sequence[str] = (),
    run: bool = False,
    validate: bool = False,
    image_root: Path | None = None,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    head_key: str = "auto",
    head_prefix: str = "auto",
    val_batch_size: int = 64,
    min_delta_macro_f1: float = 0.0,
    min_per_class_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not checkpoints:
        raise ValueError("all-fold rebalance requires at least one checkpoint")
    fold_manifest = _load_fold_manifest_summary(rebalance_manifest_summary)
    checkpoint_by_fold: dict[int, Path] = {}
    for checkpoint in checkpoints:
        fold = _checkpoint_fold_from_path(checkpoint)
        if fold in checkpoint_by_fold:
            raise ValueError(f"duplicate checkpoint for fold {fold}")
        if fold not in fold_manifest:
            raise ValueError(f"rebalance manifest summary has no entry for checkpoint fold {fold}")
        checkpoint_by_fold[fold] = checkpoint

    candidates_by_fold: dict[int, list[RebalanceCandidate]] = {}
    fold_payloads: list[dict[str, object]] = []
    for fold in sorted(checkpoint_by_fold):
        checkpoint = checkpoint_by_fold[fold]
        manifest = fold_manifest[fold]
        fold_output_dir = output_dir / f"fold{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        candidates = run_rebalance_grid(
            checkpoint=checkpoint,
            output_dir=fold_output_dir,
            tau_values=tau_values,
            crt_sampler_modes=crt_sampler_modes,
            lws_sampler_modes=lws_sampler_modes,
            train_csv=Path(str(manifest["train_csv"])),
            image_root=image_root,
            run=run,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            head_key=head_key,
            head_prefix=head_prefix,
            rebalance_manifest_summary=rebalance_manifest_summary,
        )
        fold_candidates: list[RebalanceCandidate]
        if validate:
            baseline = validate_rebalance_grid(
                checkpoint=checkpoint,
                candidates=candidates,
                output_dir=fold_output_dir,
                val_csv=Path(str(manifest["val_csv"])),
                val_image_root=image_root,
                batch_size=val_batch_size,
                device=device,
            )
            fold_candidates = [baseline, *candidates]
        else:
            fold_candidates = candidates
        candidates_by_fold[fold] = fold_candidates
        fold_payloads.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint),
                "train_csv": str(manifest["train_csv"]),
                "val_csv": str(manifest["val_csv"]),
                "output_dir": str(fold_output_dir),
                "candidates": [
                    candidate.__dict__ | {
                        "checkpoint": str(candidate.checkpoint),
                        "output": str(candidate.output),
                        "metrics_json": str(candidate.metrics_json) if candidate.metrics_json is not None else None,
                    }
                    for candidate in fold_candidates
                ],
            }
        )
    payload: dict[str, object] = {"folds": fold_payloads}
    if validate:
        aggregate = summarize_fold_rebalance_grid(
            candidates_by_fold,
            min_delta_macro_f1=min_delta_macro_f1,
            min_per_class_f1=min_per_class_f1,
            tie_epsilon=tie_epsilon,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "classifier_rebalance_all_folds_summary.json"
        json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        payload["aggregate"] = {**aggregate, "json": str(json_path)}
    return payload


def run_rebalance_grid(
    checkpoint: Path,
    output_dir: Path,
    tau_values: Sequence[float],
    crt_sampler_modes: Sequence[str],
    lws_sampler_modes: Sequence[str] = (),
    train_csv: Path | None = None,
    train_dir: Path | None = None,
    image_root: Path | None = None,
    run: bool = False,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    head_key: str = "auto",
    head_prefix: str = "auto",
    rebalance_manifest_summary: Path | None = None,
) -> list[RebalanceCandidate]:
    candidates = build_rebalance_grid(checkpoint, output_dir, tau_values, crt_sampler_modes, lws_sampler_modes)
    if not run:
        return candidates
    if any(candidate.method in {"crt", "lws"} for candidate in candidates) and rebalance_manifest_summary is None:
        raise ValueError("--run with cRT/LWS requires --rebalance-manifest-summary from --prepare-folds")
    for candidate in candidates:
        if candidate.method == "tau_norm":
            assert candidate.tau is not None
            rebalance_checkpoint(checkpoint, candidate.output, tau=candidate.tau, head_key=head_key)
        elif candidate.method == "crt":
            retrain_classifier_head(
                checkpoint_path=checkpoint,
                output_path=candidate.output,
                train_csv=train_csv,
                train_dir=train_dir,
                image_root=image_root,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                sampler_mode=str(candidate.sampler_mode),
                device=device,
                head_prefix=head_prefix,
                rebalance_manifest_summary=rebalance_manifest_summary,
            )
        elif candidate.method == "lws":
            retrain_classifier_head(
                checkpoint_path=checkpoint,
                output_path=candidate.output,
                train_csv=train_csv,
                train_dir=train_dir,
                image_root=image_root,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                sampler_mode=str(candidate.sampler_mode),
                device=device,
                head_prefix=head_prefix,
                rebalance_method="lws",
                rebalance_manifest_summary=rebalance_manifest_summary,
            )
    return candidates


def _write_metrics(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def validate_rebalance_grid(
    checkpoint: Path,
    candidates: Sequence[RebalanceCandidate],
    output_dir: Path,
    val_dir: Path | None = None,
    val_csv: Path | None = None,
    val_image_root: Path | None = None,
    batch_size: int = 64,
    device: str = "auto",
) -> RebalanceCandidate:
    if val_dir is None and val_csv is None:
        raise ValueError("Set val_dir or val_csv when validating rebalance grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = RebalanceCandidate(
        name="baseline",
        method="baseline",
        checkpoint=checkpoint,
        output=checkpoint,
        metrics_json=output_dir / "baseline_metrics.json",
    )
    baseline_metrics = validate_checkpoint(
        checkpoint=checkpoint,
        val_dir=val_dir,
        val_csv=val_csv,
        image_root=val_image_root,
        batch_size=batch_size,
        device=device,
    )
    _write_metrics(baseline.metrics_json, baseline_metrics)
    for candidate in candidates:
        candidate.metrics_json = output_dir / f"{candidate.name}_metrics.json"
        metrics = validate_checkpoint(
            checkpoint=candidate.output,
            val_dir=val_dir,
            val_csv=val_csv,
            image_root=val_image_root,
            batch_size=batch_size,
            device=device,
        )
        _write_metrics(candidate.metrics_json, metrics)
    return baseline


def validate_cli_args(args: argparse.Namespace) -> None:
    prepare_folds = bool(getattr(args, "prepare_folds", False))
    raw_checkpoint = getattr(args, "checkpoint", None)
    if raw_checkpoint is None:
        checkpoints = []
    elif isinstance(raw_checkpoint, (str, Path)):
        checkpoints = [Path(raw_checkpoint)]
    else:
        checkpoints = list(raw_checkpoint)
    if not prepare_folds and not checkpoints:
        raise ValueError("--checkpoint is required unless --prepare-folds is set")
    all_folds = len(checkpoints) > 1
    if prepare_folds:
        mixed_options = (
            bool(getattr(args, "run", False))
            or bool(getattr(args, "validate", False))
            or getattr(args, "baseline_metrics", None) is not None
            or bool(getattr(args, "candidate_metrics", None))
            or getattr(args, "checkpoint", None) is not None
            or getattr(args, "val_dir", None) is not None
            or getattr(args, "val_csv", None) is not None
            or getattr(args, "rebalance_manifest_summary", None) is not None
        )
        if mixed_options:
            raise ValueError("--prepare-folds cannot be combined with grid runtime options")
    if all_folds and getattr(args, "baseline_metrics", None) is not None:
        raise ValueError("all-fold rebalance uses automatic validation; do not pass --baseline-metrics")
    if all_folds and getattr(args, "candidate_metrics", None):
        raise ValueError("all-fold rebalance uses automatic validation; do not pass --candidate-metrics")
    if all_folds and getattr(args, "train_csv", None) is not None:
        raise ValueError("all-fold rebalance reads per-fold train_csv from --rebalance-manifest-summary")
    if all_folds and not getattr(args, "rebalance_manifest_summary", None):
        raise ValueError("all-fold rebalance requires --rebalance-manifest-summary")
    if all_folds and not getattr(args, "validate", False):
        raise ValueError("all-fold rebalance requires --validate so candidates are selected by cross-fold metrics")
    if args.validate:
        if args.baseline_metrics is not None or args.candidate_metrics:
            raise ValueError("Do not combine --validate with --baseline-metrics or --candidate-metrics")
        if not all_folds and args.val_dir is None and args.val_csv is None:
            raise ValueError("--validate requires --val-dir or --val-csv")
    if (
        not prepare_folds
        and getattr(args, "run", False)
        and (getattr(args, "crt_sampler_mode", None) or getattr(args, "lws_sampler_mode", None))
        and getattr(args, "rebalance_manifest_summary", None) is None
    ):
        raise ValueError("--run with cRT/LWS requires --rebalance-manifest-summary from --prepare-folds")
    if prepare_folds:
        if args.train_csv is None and args.train_dir is None:
            raise ValueError("--prepare-folds requires --train-csv or --train-dir")
        if args.train_csv is not None and args.class_map is None:
            raise ValueError("--prepare-folds with --train-csv requires --class-map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and summarize tau/crt classifier rebalance grids.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--crt-sampler-mode", nargs="*", default=["sqrt", "class_balanced"])
    parser.add_argument("--lws-sampler-mode", nargs="*", default=[])
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--prepare-folds", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--head-key", type=str, default="auto")
    parser.add_argument("--head-prefix", type=str, default="auto")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--val-image-root", type=Path, default=None)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--candidate-metrics", type=Path, nargs="*", default=None)
    parser.add_argument("--rebalance-manifest-summary", type=Path, default=None)
    parser.add_argument("--min-delta-macro-f1", type=float, default=0.0)
    parser.add_argument("--min-per-class-f1", type=float, default=0.0)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    if args.prepare_folds:
        payload = build_fold_safe_rebalance_manifests(
            train_csv=args.train_csv,
            train_dir=args.train_dir,
            image_root=args.image_root,
            class_map=args.class_map,
            output_dir=args.output_dir / "fold_safe_manifests",
            folds=args.folds,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    checkpoints = list(args.checkpoint or [])
    if not checkpoints:
        raise ValueError("--checkpoint is required unless --prepare-folds is set")
    if len(checkpoints) > 1:
        payload = run_all_fold_rebalance_grid(
            checkpoints=checkpoints,
            rebalance_manifest_summary=args.rebalance_manifest_summary,
            output_dir=args.output_dir,
            tau_values=args.tau,
            crt_sampler_modes=args.crt_sampler_mode,
            lws_sampler_modes=args.lws_sampler_mode,
            run=args.run,
            validate=args.validate,
            image_root=args.image_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            head_key=args.head_key,
            head_prefix=args.head_prefix,
            val_batch_size=args.val_batch_size,
            min_delta_macro_f1=args.min_delta_macro_f1,
            min_per_class_f1=args.min_per_class_f1,
            tie_epsilon=args.tie_epsilon,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    checkpoint = checkpoints[0]
    candidates = run_rebalance_grid(
        checkpoint=checkpoint,
        output_dir=args.output_dir,
        tau_values=args.tau,
        crt_sampler_modes=args.crt_sampler_mode,
        lws_sampler_modes=args.lws_sampler_mode,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        image_root=args.image_root,
        run=args.run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        head_key=args.head_key,
        head_prefix=args.head_prefix,
        rebalance_manifest_summary=args.rebalance_manifest_summary,
    )
    payload: dict[str, object] = {
        "candidates": [candidate.__dict__ | {"checkpoint": str(candidate.checkpoint), "output": str(candidate.output)} for candidate in candidates]
    }
    baseline_metrics = args.baseline_metrics
    if args.validate:
        baseline = validate_rebalance_grid(
            checkpoint=checkpoint,
            candidates=candidates,
            output_dir=args.output_dir,
            val_dir=args.val_dir,
            val_csv=args.val_csv,
            val_image_root=args.val_image_root,
            batch_size=args.val_batch_size,
            device=args.device,
        )
        baseline_metrics = baseline.metrics_json
    if baseline_metrics is not None or args.candidate_metrics or args.validate:
        baseline = RebalanceCandidate(
            name="baseline",
            method="baseline",
            checkpoint=checkpoint,
            output=checkpoint,
            metrics_json=baseline_metrics,
        )
        if baseline.metrics_json is None:
            raise ValueError("--baseline-metrics is required when summarizing candidate metrics")
        if args.candidate_metrics is not None:
            if len(args.candidate_metrics) != len(candidates):
                raise ValueError("--candidate-metrics count must match generated candidate count")
            for candidate, metrics_path in zip(candidates, args.candidate_metrics):
                candidate.metrics_json = metrics_path
        summary = summarize_rebalance_grid(
            [baseline, *candidates],
            min_delta_macro_f1=args.min_delta_macro_f1,
            min_per_class_f1=args.min_per_class_f1,
            tie_epsilon=args.tie_epsilon,
        )
        json_path, csv_path = write_summary(args.output_dir, summary)
        payload = {**summary, "json": str(json_path), "csv": str(csv_path)}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
