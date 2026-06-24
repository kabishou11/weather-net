from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import ManifestRow


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding vectors must have finite non-zero norm")
    return vector / norm


def _row_image_id(row: ManifestRow) -> str:
    return row.image_id or row.path.name


def load_embedding_index(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "image_id" not in data.files or "embedding" not in data.files:
        raise ValueError(f"Embedding file must contain image_id and embedding arrays: {path}")
    image_ids = [str(value) for value in np.asarray(data["image_id"], dtype=object).tolist()]
    embeddings = np.asarray(data["embedding"], dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Embedding array must be 2D: {path}")
    if len(image_ids) != embeddings.shape[0]:
        raise ValueError(f"image_id and embedding row counts must match: {path}")
    if embeddings.shape[1] == 0:
        raise ValueError(f"Embedding dimension must be positive: {path}")
    index: dict[str, np.ndarray] = {}
    for image_id, vector in zip(image_ids, embeddings):
        if image_id in index:
            raise ValueError(f"Duplicate embedding image_id: {image_id}")
        index[image_id] = _normalize_vector(vector)
    return index


def build_class_prototypes(
    rows: Sequence[ManifestRow],
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    missing: list[str] = []
    for row in rows:
        if row.source == "pseudo":
            continue
        if row.label_name is None:
            raise ValueError(f"Labeled row is missing label_name: {row.path}")
        image_id = _row_image_id(row)
        vector = embeddings.get(image_id)
        if vector is None:
            missing.append(image_id)
            continue
        grouped.setdefault(row.label_name, []).append(_normalize_vector(vector))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing embeddings for labeled rows: {preview}")
    if not grouped:
        raise ValueError("No labeled embeddings available for prototypes")
    return {
        label: _normalize_vector(np.mean(vectors, axis=0))
        for label, vectors in grouped.items()
    }


def _read_pseudo_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "label", "confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pseudo label CSV missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _prototype_scores(vector: np.ndarray, prototypes: Mapping[str, np.ndarray]) -> dict[str, float]:
    vector = _normalize_vector(vector)
    expected_dim = len(next(iter(prototypes.values())))
    if len(vector) != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}")
    return {label: float(np.dot(vector, prototype)) for label, prototype in prototypes.items()}


def filter_pseudo_labels_with_embedding_guard(
    train_rows: Sequence[ManifestRow],
    pseudo_csv: Path,
    embeddings: Mapping[str, np.ndarray],
    output_csv: Path,
    rejected_csv: Path,
    min_similarity: float = 0.8,
    min_margin: float = 0.1,
) -> dict[str, int]:
    if not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be in [-1, 1]")
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")
    prototypes = build_class_prototypes(train_rows, embeddings)
    pseudo_rows = _read_pseudo_rows(pseudo_csv)
    unknown_labels = sorted({row["label"].strip() for row in pseudo_rows} - set(prototypes))
    if unknown_labels:
        raise ValueError(f"Pseudo labels contain unknown labels: {unknown_labels}")

    kept: list[tuple[dict[str, str], float, float]] = []
    rejected: list[tuple[dict[str, str], float, float, str]] = []
    missing: list[str] = []
    for row in pseudo_rows:
        image_id = row["image"].strip()
        label = row["label"].strip()
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Pseudo label confidence must be finite and in [0, 1]: {row['confidence']}")
        vector = embeddings.get(image_id)
        if vector is None:
            missing.append(image_id)
            continue
        scores = _prototype_scores(vector, prototypes)
        target_score = scores[label]
        competitor_scores = [score for other_label, score in scores.items() if other_label != label]
        best_competitor = max(competitor_scores) if competitor_scores else -1.0
        margin = target_score - best_competitor
        if target_score < min_similarity:
            rejected.append((row, target_score, margin, "low_similarity"))
        elif margin < min_margin:
            rejected.append((row, target_score, margin, "low_margin"))
        else:
            kept.append((row, target_score, margin))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing embeddings for pseudo rows: {preview}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "confidence", "semantic_similarity", "semantic_margin"])
        for row, similarity, margin in kept:
            writer.writerow(
                [
                    row["image"].strip(),
                    row["label"].strip(),
                    f"{float(row['confidence']):.6f}",
                    f"{similarity:.6f}",
                    f"{margin:.6f}",
                ]
            )

    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    with rejected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "confidence", "semantic_similarity", "semantic_margin", "reason"])
        for row, similarity, margin, reason in rejected:
            writer.writerow(
                [
                    row["image"].strip(),
                    row["label"].strip(),
                    f"{float(row['confidence']):.6f}",
                    f"{similarity:.6f}",
                    f"{margin:.6f}",
                    reason,
                ]
            )

    rejection_counts = Counter(reason for _row, _similarity, _margin, reason in rejected)
    return {
        "input": len(pseudo_rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_low_similarity": rejection_counts["low_similarity"],
        "rejected_low_margin": rejection_counts["low_margin"],
    }
