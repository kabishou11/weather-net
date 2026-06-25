from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference_budget import check_inference_budget
from src.weather_net.config import load_config, validate_config
from src.weather_net.postprocess import load_decision_params
from training_preflight import run_preflight_checks


def _violation(gate: str, message: str) -> dict[str, str]:
    return {"gate": gate, "message": message}


def _pass(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "pass", **(payload or {})}


def _fail(message: str) -> dict[str, Any]:
    return {"status": "fail", "message": message}


def _load_json_object(path: Path, gate: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{gate} file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{gate} file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{gate} file must contain a JSON object: {path}")
    return payload


def _check_nested_oof_decision(decision_dir: Path, require_nested_oof: bool) -> dict[str, Any]:
    decision_path = decision_dir / "decision_params.json"
    if not decision_path.exists():
        raise ValueError(f"decision_params.json is required: {decision_path}")
    params = load_decision_params(decision_path)
    payload: dict[str, Any] = {
        "decision_params": str(decision_path),
        "class_names": params.class_names,
        "checkpoint_count": len(params.checkpoints or params.weights or []),
        "scores": params.scores,
    }
    if not require_nested_oof:
        return payload

    nested_path = decision_dir / "nested_oof_scorecard.json"
    nested = _load_json_object(nested_path, "nested_oof_scorecard")
    if "pooled_scores" not in nested:
        raise ValueError("nested_oof_scorecard.json is missing pooled_scores")
    if "bias_accepted" not in nested:
        raise ValueError("nested_oof_scorecard.json is missing bias_accepted")
    scores = params.scores
    if "nested_bias_accepted" not in scores:
        raise ValueError("decision_params.json scores are missing nested_bias_accepted")
    payload.update(
        {
            "nested_oof_scorecard": str(nested_path),
            "nested_bias_accepted": float(scores["nested_bias_accepted"]),
            "final_bias_accepted": float(scores.get("bias_accepted", 0.0)),
            "nested_pooled_scores": nested["pooled_scores"],
        }
    )
    return payload


def run_readiness_checks(
    config_path: Path,
    train_csv: Path | None = None,
    train_dir: Path | None = None,
    image_root: Path | None = None,
    class_map: Path | None = None,
    expected_classes: list[str] | None = None,
    require_all_expected_classes: bool = False,
    allow_external_data: bool = False,
    external_max_ratio: float | None = None,
    external_max_sample_weight: float | None = None,
    pseudo_min_confidence: float | None = None,
    pseudo_max_ratio: float | None = None,
    allow_pseudo_teacher_distillation: bool = False,
    min_images_per_class: int = 1,
    min_labeled_images_per_class: int | None = None,
    inference_stats: Path | None = None,
    max_seconds_per_image: float = 0.05,
    max_checkpoints: int = 1,
    allow_tta: bool = False,
    decision_dir: Path | None = None,
    require_nested_oof: bool = False,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    violations: list[dict[str, str]] = []

    try:
        config = load_config(config_path)
        if train_csv is not None:
            config.data.train_csv = train_csv
        if train_dir is not None:
            config.data.train_dir = train_dir
        if image_root is not None:
            config.data.image_root = image_root
        if class_map is not None:
            config.data.class_map = class_map
        validate_config(config)
        preflight = run_preflight_checks(
            config_path=config_path,
            config=config,
            profile="server-strict",
            train_csv=config.data.train_csv,
            train_dir=config.data.train_dir,
            image_root=config.data.image_root,
            class_map=config.data.class_map,
            expected_classes=expected_classes,
            require_all_expected_classes=require_all_expected_classes,
            allow_external_data=allow_external_data,
            external_max_ratio=external_max_ratio,
            external_max_sample_weight=external_max_sample_weight,
            pseudo_min_confidence=pseudo_min_confidence,
            pseudo_max_ratio=pseudo_max_ratio,
            allow_pseudo_teacher_distillation=allow_pseudo_teacher_distillation,
            min_images_per_class=min_images_per_class,
            min_labeled_images_per_class=min_labeled_images_per_class,
            inference_stats=None,
        )
        gates["server_strict_preflight"] = _pass(preflight)
    except Exception as error:
        message = str(error)
        gates["server_strict_preflight"] = _fail(message)
        violations.append(_violation("server_strict_preflight", message))

    if decision_dir is not None:
        try:
            gates["nested_oof_decision"] = _pass(
                _check_nested_oof_decision(decision_dir, require_nested_oof=require_nested_oof)
            )
        except Exception as error:
            message = str(error)
            gates["nested_oof_decision"] = _fail(message)
            violations.append(_violation("nested_oof_decision", message))
    elif require_nested_oof:
        message = "--decision-dir is required when --require-nested-oof is set"
        gates["nested_oof_decision"] = _fail(message)
        violations.append(_violation("nested_oof_decision", message))

    if inference_stats is not None:
        try:
            result = check_inference_budget(
                stats_path=inference_stats,
                max_seconds_per_image=max_seconds_per_image,
                max_checkpoints=max_checkpoints,
                allow_tta=allow_tta,
            )
            gates["inference_budget"] = result
            if result["status"] != "pass":
                violations.append(
                    _violation("inference_budget", f"inference budget failed: {result['violations']}")
                )
        except Exception as error:
            message = str(error)
            gates["inference_budget"] = _fail(message)
            violations.append(_violation("inference_budget", message))

    status = "pass" if not violations else "fail"
    return {
        "status": status,
        "config": str(config_path),
        "gates": gates,
        "violations": violations,
        "recommendation": (
            "ready_for_server_training"
            if status == "pass"
            else "fix_readiness_violations_before_server_training"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed readiness gate before expensive server training.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnextv2_384.yaml"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--expected-classes", nargs="+", default=None)
    parser.add_argument("--require-all-expected-classes", action="store_true")
    parser.add_argument("--allow-external-data", action="store_true")
    parser.add_argument("--external-max-ratio", type=float, default=None)
    parser.add_argument("--external-max-sample-weight", type=float, default=None)
    parser.add_argument("--pseudo-min-confidence", type=float, default=None)
    parser.add_argument("--pseudo-max-ratio", type=float, default=None)
    parser.add_argument("--allow-pseudo-teacher-distillation", action="store_true")
    parser.add_argument("--min-images-per-class", type=int, default=1)
    parser.add_argument("--min-labeled-images-per-class", type=int, default=None)
    parser.add_argument("--inference-stats", type=Path, default=None)
    parser.add_argument("--max-seconds-per-image", type=float, default=0.05)
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--allow-tta", action="store_true")
    parser.add_argument("--decision-dir", type=Path, default=None)
    parser.add_argument("--require-nested-oof", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_readiness_checks(
        config_path=args.config,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        image_root=args.image_root,
        class_map=args.class_map,
        expected_classes=args.expected_classes,
        require_all_expected_classes=args.require_all_expected_classes,
        allow_external_data=args.allow_external_data,
        external_max_ratio=args.external_max_ratio,
        external_max_sample_weight=args.external_max_sample_weight,
        pseudo_min_confidence=args.pseudo_min_confidence,
        pseudo_max_ratio=args.pseudo_max_ratio,
        allow_pseudo_teacher_distillation=args.allow_pseudo_teacher_distillation,
        min_images_per_class=args.min_images_per_class,
        min_labeled_images_per_class=args.min_labeled_images_per_class,
        inference_stats=args.inference_stats,
        max_seconds_per_image=args.max_seconds_per_image,
        max_checkpoints=args.max_checkpoints,
        allow_tta=args.allow_tta,
        decision_dir=args.decision_dir,
        require_nested_oof=args.require_nested_oof,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
