from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge labeled training data with pseudo labels.")
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--pseudo-csv", type=Path, required=True)
    parser.add_argument("--pseudo-image-root", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--allow-pseudo-teacher", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("merged_train.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = merge_training_with_pseudo_labels(
        train_dir=args.train_dir,
        train_csv=args.train_csv,
        image_root=args.image_root,
        pseudo_csv=args.pseudo_csv,
        pseudo_image_root=args.pseudo_image_root,
        output_csv=args.output,
        min_confidence=args.min_confidence,
        allow_pseudo_teacher=args.allow_pseudo_teacher,
    )
    print(json.dumps({**stats, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
