from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch


HEAD_KEY_CANDIDATES = (
    "classifier.weight",
    "head.weight",
    "fc.weight",
    "module.classifier.weight",
    "module.head.weight",
    "module.fc.weight",
)


def _resolve_head_key(state_dict: Mapping[str, torch.Tensor], head_key: str) -> str:
    if head_key != "auto":
        if head_key not in state_dict:
            raise ValueError(f"classifier head not found: {head_key}")
        return head_key
    matches = [key for key in HEAD_KEY_CANDIDATES if key in state_dict]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple classifier head candidates found: {matches}; pass --head-key")
    suffix_matches = [
        key
        for key, value in state_dict.items()
        if key.endswith((".classifier.weight", ".head.weight", ".fc.weight"))
        and isinstance(value, torch.Tensor)
        and value.ndim == 2
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise ValueError(f"Multiple classifier head candidates found: {suffix_matches}; pass --head-key")
    raise ValueError("classifier head weight not found; pass --head-key")


def apply_tau_norm_to_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    tau: float,
    head_key: str = "auto",
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    resolved_key = _resolve_head_key(state_dict, head_key)
    weight = state_dict[resolved_key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("classifier head weight must be a 2D tensor")
    output = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}
    norms = weight.float().norm(dim=1, keepdim=True).clamp_min(eps)
    output[resolved_key] = (weight.float() / norms.pow(float(tau))).to(dtype=weight.dtype)
    return output


def rebalance_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    tau: float = 1.0,
    head_key: str = "auto",
    map_location: str = "cpu",
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "model_state" not in checkpoint or not isinstance(checkpoint["model_state"], Mapping):
        raise ValueError("checkpoint must contain a model_state mapping")
    resolved_key = _resolve_head_key(checkpoint["model_state"], head_key)
    checkpoint = dict(checkpoint)
    checkpoint["model_state"] = apply_tau_norm_to_state_dict(
        checkpoint["model_state"],
        tau=tau,
        head_key=resolved_key,
    )
    checkpoint["rebalance"] = {
        "method": "tau_norm",
        "tau": float(tau),
        "head_key": resolved_key,
        "source_checkpoint": str(checkpoint_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "method": "tau_norm",
        "tau": float(tau),
        "head_key": resolved_key,
        "source": str(checkpoint_path),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply classifier rebalancing to a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["tau_norm"], default="tau_norm")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--head-key", type=str, default="auto")
    parser.add_argument("--map-location", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = rebalance_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tau=args.tau,
        head_key=args.head_key,
        map_location=args.map_location,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
