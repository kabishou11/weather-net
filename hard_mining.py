from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.hard_mining import apply_hard_mining_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hard-sample weighted training CSV from OOF predictions.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("train_hard_weighted.csv"))
    parser.add_argument("--hard-output", type=Path, default=Path("hard_samples.csv"))
    parser.add_argument("--error-boost", type=float, default=1.0)
    parser.add_argument("--low-margin-boost", type=float, default=0.5)
    parser.add_argument("--high-loss-boost", type=float, default=0.5)
    parser.add_argument("--low-margin-threshold", type=float, default=0.1)
    parser.add_argument("--high-loss-quantile", type=float, default=0.75)
    parser.add_argument("--max-weight", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = apply_hard_mining_weights(
        train_csv=args.train_csv,
        oof_csv=args.oof_csv,
        output_csv=args.output,
        hard_csv=args.hard_output,
        error_boost=args.error_boost,
        low_margin_boost=args.low_margin_boost,
        high_loss_boost=args.high_loss_boost,
        low_margin_threshold=args.low_margin_threshold,
        high_loss_quantile=args.high_loss_quantile,
        max_weight=args.max_weight,
    )
    print(json.dumps({**stats, "output": str(args.output), "hard_output": str(args.hard_output)}, indent=2))


if __name__ == "__main__":
    main()
