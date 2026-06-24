from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stats file must contain a JSON object: {path}")
    return payload


def check_inference_budget(
    stats_path: Path,
    max_seconds_per_image: float,
    max_checkpoints: int = 1,
    allow_tta: bool = False,
) -> dict[str, object]:
    if max_seconds_per_image <= 0:
        raise ValueError("max_seconds_per_image must be positive")
    if max_checkpoints <= 0:
        raise ValueError("max_checkpoints must be positive")
    stats = _load_stats(stats_path)
    images = int(stats.get("images", 0))
    seconds = float(stats.get("seconds", 0.0))
    if images <= 0:
        raise ValueError("stats images must be positive")
    if seconds <= 0:
        raise ValueError("stats seconds must be positive")
    checkpoints = [str(path) for path in stats.get("checkpoints", [])]
    if not checkpoints:
        raise ValueError("stats checkpoints must be non-empty")

    seconds_per_image = seconds / images
    violations: list[str] = []
    if seconds_per_image > max_seconds_per_image:
        violations.append("seconds_per_image")
    if len(checkpoints) > max_checkpoints:
        violations.append("checkpoints")
    if bool(stats.get("tta", False)) and not allow_tta:
        violations.append("tta")
    status = "pass" if not violations else "fail"
    return {
        "status": status,
        "stats": str(stats_path),
        "images": images,
        "seconds": seconds,
        "seconds_per_image": seconds_per_image,
        "images_per_second": images / seconds,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "tta": bool(stats.get("tta", False)),
        "amp": bool(stats.get("amp", False)),
        "budget": {
            "max_seconds_per_image": float(max_seconds_per_image),
            "max_checkpoints": int(max_checkpoints),
            "allow_tta": bool(allow_tta),
        },
        "violations": violations,
        "recommendation": "submission_ready" if status == "pass" else "prefer_soup_or_fast_single_model",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check inference stats against a submission speed budget.")
    parser.add_argument("--stats", type=Path, required=True, help="Path to infer.py .stats.json output.")
    parser.add_argument("--max-seconds-per-image", type=float, required=True)
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--allow-tta", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_inference_budget(
        stats_path=args.stats,
        max_seconds_per_image=args.max_seconds_per_image,
        max_checkpoints=args.max_checkpoints,
        allow_tta=args.allow_tta,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
