import json
import sys
from pathlib import Path

import yaml


def test_build_confusion_hard_ablation_configs_generates_weighted_csv_and_configs(tmp_path: Path) -> None:
    from confusion_hard_ablation import build_confusion_hard_ablation_configs

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source\n"
        "rain_1.jpg,rain,labeled\n"
        "rain_2.jpg,rain,labeled\n"
        "sunny_1.jpg,sunny,labeled\n",
        encoding="utf-8",
    )
    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "rain_1.jpg,rain,fog,False,0.900000,0.700000,2.000000\n"
        "rain_2.jpg,rain,fog,False,0.850000,0.650000,1.900000\n"
        "sunny_1.jpg,sunny,sunny,True,0.990000,0.900000,0.010000\n",
        encoding="utf-8",
    )

    outputs = build_confusion_hard_ablation_configs(
        base_config_path=Path("configs/convnextv2_384_hard_finetune.yaml"),
        train_csv=train_csv,
        oof_csv=oof_csv,
        output_root=tmp_path / "ablation",
        pair_confusion_boosts=[0.0, 0.4],
        pair_min_support=2,
        pair_min_error_share=0.35,
        max_weight=2.2,
        folds=3,
        epochs=2,
    )

    assert [item.name for item in outputs] == ["pair0p00", "pair0p40"]
    weighted_rows = (tmp_path / "ablation" / "pair0p40" / "train_confusion_hard_weighted.csv").read_text(
        encoding="utf-8"
    )
    assert "2.200000" in weighted_rows
    hard_rows = (tmp_path / "ablation" / "pair0p40" / "confusion_hard_samples.csv").read_text(encoding="utf-8")
    assert "pair_confusion:rain->fog" in hard_rows

    payload = yaml.safe_load(outputs[1].config_path.read_text(encoding="utf-8"))
    assert payload["data"]["train_csv"] == str(tmp_path / "ablation" / "pair0p40" / "train_confusion_hard_weighted.csv")
    assert payload["data"]["folds"] == 3
    assert payload["train"]["epochs"] == 2
    assert payload["train"]["sample_weight_usage"] == "sampler"
    assert payload["train"]["sampler_mode"] == "sample_weighted"
    assert payload["train"]["output_dir"] == str(tmp_path / "ablation" / "pair0p40" / "train")


def test_summarize_confusion_hard_ablation_selects_stable_winner(tmp_path: Path) -> None:
    from confusion_hard_ablation import summarize_confusion_hard_ablation

    for name, macro_f1, fog_f1, pair_errors in [
        ("pair0p00", 0.740, 0.690, {"rain->fog": 9}),
        ("pair0p40", 0.746, 0.700, {"rain->fog": 5}),
        ("pair0p80", 0.751, 0.500, {"rain->fog": 4}),
    ]:
        train_dir = tmp_path / name / "train"
        train_dir.mkdir(parents=True)
        (train_dir / "training_summary.json").write_text(
            json.dumps(
                [
                    {
                        "fold": 0,
                        "best_macro_f1": macro_f1,
                        "per_class_f1": {"fog": fog_f1, "sunny": 0.8},
                        "confusion_matrix": [[4, 1], [0, 5]],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (tmp_path / name / "confusion_hard_samples.csv").write_text(
            "image,label,sample_weight,hard_reason,confidence,margin,loss,prediction,confusion_pair\n"
            f"{name}_rain.jpg,rain,2.000000,error+pair_confusion:rain->fog,0.9,0.7,2.0,fog,rain->fog\n",
            encoding="utf-8",
        )
        (tmp_path / name / "pair_stats.json").write_text(
            json.dumps({"pair_confusion_boosted": pair_errors.get("rain->fog", 0)}, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = summarize_confusion_hard_ablation(
        output_root=tmp_path,
        names=["pair0p00", "pair0p40", "pair0p80"],
        baseline_name="pair0p00",
        min_per_class_f1=0.6,
        min_delta_macro_f1=0.003,
        tie_epsilon=0.0,
    )

    assert summary["best_name"] == "pair0p40"
    rejected = {item["name"]: item["selection_status"] for item in summary["results"]}
    assert rejected["pair0p80"] == "rejected_min_per_class_f1"
    assert summary["results"][1]["delta_macro_f1_vs_baseline"] == 0.006000000000000005
    assert summary["results"][1]["hard_confusion_pairs"] == {"rain->fog": 1}


def test_confusion_hard_ablation_parse_args_accepts_pair_sweep(monkeypatch) -> None:
    from confusion_hard_ablation import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "confusion_hard_ablation.py",
            "--train-csv",
            "data/train.csv",
            "--oof-csv",
            "outputs/oof/oof_predictions.csv",
            "--pair-confusion-boost",
            "0.0",
            "0.4",
            "--folds",
            "3",
            "--epochs",
            "2",
            "--min-delta-macro-f1",
            "0.005",
            "--summarize",
        ],
    )

    args = parse_args()

    assert args.train_csv == Path("data/train.csv")
    assert args.oof_csv == Path("outputs/oof/oof_predictions.csv")
    assert args.pair_confusion_boost == [0.0, 0.4]
    assert args.folds == 3
    assert args.epochs == 2
    assert args.min_delta_macro_f1 == 0.005
    assert args.summarize is True
