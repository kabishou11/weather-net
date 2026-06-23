from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClassificationReport:
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1

    per_class_f1: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        per_class_f1[name] = _f1_from_counts(tp, fp, fn)

    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    return ClassificationReport(
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class_f1=per_class_f1,
        confusion_matrix=matrix.astype(int).tolist(),
    )
