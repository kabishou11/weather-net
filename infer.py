from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.config import load_config, resolve_device
from src.weather_net.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather image classification inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.yaml"))
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config if args.config.exists() else None)
    device = resolve_device(args.device)

    stats = run_inference(
        checkpoints=args.checkpoint,
        output_csv=args.output or config.infer.output_csv,
        test_dir=args.test_dir or config.data.test_dir,
        test_csv=args.test_csv or config.data.test_csv,
        image_root=args.image_root or config.data.image_root,
        batch_size=args.batch_size or config.infer.batch_size,
        device=device,
        tta=args.tta or config.infer.tta,
        image_column=config.data.image_column,
        num_workers=config.data.num_workers,
        checkpoint_weights=args.weights,
        transform_backend=config.data.transform_backend,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
