from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.teacher import write_teacher_csv_from_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write OOF teacher probabilities back into a training CSV.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("train_teacher.csv"))
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--max-top1-mismatch-rate", type=float, default=None)
    parser.add_argument("--min-mean-true-probability", type=float, default=None)
    parser.add_argument("--min-samples-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = write_teacher_csv_from_oof(
        train_csv=args.train_csv,
        image_root=args.image_root,
        oof_npz=args.oof,
        output_csv=args.output,
        image_column=args.image_column,
        label_column=args.label_column,
        max_top1_mismatch_rate=args.max_top1_mismatch_rate,
        min_mean_true_probability=args.min_mean_true_probability,
        min_samples_per_class=args.min_samples_per_class,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
