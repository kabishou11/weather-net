from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.soup import save_model_soup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average same-architecture checkpoints into a model soup.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--map-location", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    soup = save_model_soup(
        checkpoints=args.checkpoints,
        output_path=args.output,
        weights=args.weights,
        map_location=args.map_location,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_name": soup["model_name"],
                "image_size": soup["image_size"],
                "num_classes": len(soup["class_to_idx"]),
                "soup": soup["soup"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
