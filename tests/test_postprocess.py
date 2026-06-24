import json
from pathlib import Path

import numpy as np
import pytest


def test_apply_decision_params_temperature_and_bias() -> None:
    from src.weather_net.postprocess import apply_decision_params

    logits = np.array([[2.0, 0.0], [0.0, 2.0]])

    adjusted = apply_decision_params(logits, temperature=2.0, bias=[0.5, -0.5])

    np.testing.assert_allclose(adjusted, np.array([[1.5, -0.5], [0.5, 0.5]]))


def test_apply_decision_params_rejects_bad_temperature_and_bias() -> None:
    from src.weather_net.postprocess import apply_decision_params

    logits = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError, match="temperature"):
        apply_decision_params(logits, temperature=0.0)
    with pytest.raises(ValueError, match="bias"):
        apply_decision_params(logits, bias=[0.1, 0.2, 0.3])


def test_macro_f1_from_logits_matches_classification_report() -> None:
    from src.weather_net.metrics import classification_report
    from src.weather_net.postprocess import macro_f1_from_logits

    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.2, 0.3]])
    y_true = np.array([0, 1, 0])
    class_names = ["rain", "sunny"]

    score = macro_f1_from_logits(logits, y_true, class_names)
    expected = classification_report(y_true, [0, 1, 1], class_names).macro_f1

    assert score == expected


def test_search_per_class_bias_can_improve_macro_f1() -> None:
    from src.weather_net.postprocess import macro_f1_from_logits, search_per_class_bias

    logits = np.array(
        [
            [2.0, 0.0],
            [1.0, 0.8],
            [0.9, 0.7],
            [0.1, 1.2],
        ]
    )
    y_true = np.array([0, 1, 1, 1])
    class_names = ["rain", "sunny"]

    baseline = macro_f1_from_logits(logits, y_true, class_names)
    result = search_per_class_bias(
        logits,
        y_true,
        class_names,
        steps=[0.5, 0.25],
        max_passes=4,
        max_abs_bias=2.0,
    )

    assert result.score >= baseline
    assert abs(sum(result.bias)) < 1e-9
    assert result.bias[1] > result.bias[0]


def test_search_temperature_prefers_one_on_tied_macro_f1() -> None:
    from src.weather_net.postprocess import search_temperature

    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    y_true = np.array([0, 1])

    temperature, score = search_temperature(
        logits,
        y_true,
        ["rain", "sunny"],
        candidates=[0.75, 1.25, 1.0],
    )

    assert temperature == 1.0
    assert score == 1.0


def test_greedy_search_ensemble_weights_keeps_or_improves_best_model() -> None:
    from src.weather_net.postprocess import greedy_search_ensemble_weights, macro_f1_from_logits

    model_a = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    model_b = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    logits_by_model = np.stack([model_a, model_b], axis=0)
    y_true = np.array([0, 1, 1, 0])
    class_names = ["rain", "sunny"]

    best_single = max(macro_f1_from_logits(model, y_true, class_names) for model in logits_by_model)
    result = greedy_search_ensemble_weights(logits_by_model, y_true, class_names, max_steps=4)

    assert result.score >= best_single
    assert pytest.approx(sum(result.weights), abs=1e-9) == 1.0
    assert len(result.weights) == 2


def test_bootstrap_macro_f1_summary_reports_mean_and_worst_quantile() -> None:
    from src.weather_net.postprocess import bootstrap_macro_f1_summary

    base_logits = np.array(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    improved_logits = base_logits.copy()
    improved_logits[2] = [0.0, 3.0]
    y_true = np.array([0, 1, 1, 1])

    summary = bootstrap_macro_f1_summary(
        baseline_logits=base_logits,
        candidate_logits=improved_logits,
        y_true=y_true,
        class_names=["rain", "sunny"],
        rounds=64,
        sample_fraction=0.75,
        seed=7,
    )

    assert summary["rounds"] == 64
    assert summary["stratified"] == 1
    assert summary["missing_class_rounds"] == 0
    assert summary["candidate_mean_macro_f1"] >= summary["baseline_mean_macro_f1"]
    assert summary["delta_mean_macro_f1"] >= 0
    assert summary["delta_q05_macro_f1"] >= 0
    assert 0 <= summary["candidate_q05_macro_f1"] <= 1


def test_bootstrap_macro_f1_summary_keeps_tail_classes_when_stratified() -> None:
    from src.weather_net.postprocess import bootstrap_macro_f1_summary

    logits = np.array(
        [
            [3.0, 0.0],
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 0, 0, 1])

    summary = bootstrap_macro_f1_summary(
        baseline_logits=logits,
        candidate_logits=logits,
        y_true=y_true,
        class_names=["rain", "sunny"],
        rounds=32,
        sample_fraction=0.5,
        seed=3,
        stratified=True,
    )

    assert summary["stratified"] == 1
    assert summary["missing_class_rounds"] == 0
    assert summary["sample_size"] >= 2


def test_bootstrap_macro_f1_summary_rejects_bad_arguments() -> None:
    from src.weather_net.postprocess import bootstrap_macro_f1_summary

    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])

    with pytest.raises(ValueError, match="rounds"):
        bootstrap_macro_f1_summary(logits, logits, y_true, ["rain", "sunny"], rounds=0)
    with pytest.raises(ValueError, match="sample_fraction"):
        bootstrap_macro_f1_summary(logits, logits, y_true, ["rain", "sunny"], sample_fraction=0.0)
    with pytest.raises(ValueError, match="shape"):
        bootstrap_macro_f1_summary(logits, logits[:1], y_true, ["rain", "sunny"])
    with pytest.raises(ValueError, match="quantile"):
        bootstrap_macro_f1_summary(logits, logits, y_true, ["rain", "sunny"], quantile=-0.1)
    with pytest.raises(ValueError, match="empty"):
        bootstrap_macro_f1_summary(logits[:0], logits[:0], y_true[:0], ["rain", "sunny"])


def test_decision_params_round_trip_and_class_validation(tmp_path: Path) -> None:
    from src.weather_net.postprocess import load_decision_params, write_decision_params

    path = tmp_path / "decision_params.json"
    write_decision_params(
        path,
        class_names=["rain", "sunny"],
        temperature=1.25,
        bias=[0.1, -0.1],
        weights=[0.75, 0.25],
        scores={"baseline_macro_f1": 0.5, "tuned_macro_f1": 0.75},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    params = load_decision_params(path, expected_class_names=["rain", "sunny"])

    assert params.temperature == 1.25
    assert params.bias == [0.1, -0.1]
    assert params.weights == [0.75, 0.25]
    assert params.checkpoints is None

    with pytest.raises(ValueError, match="class_names"):
        load_decision_params(path, expected_class_names=["sunny", "rain"])


def test_decision_params_can_bind_checkpoint_order(tmp_path: Path) -> None:
    from src.weather_net.postprocess import load_decision_params, write_decision_params

    path = tmp_path / "decision_params.json"
    write_decision_params(
        path,
        class_names=["rain", "sunny"],
        temperature=1.0,
        bias=[0.0, 0.0],
        weights=[0.8, 0.2],
        scores={},
        checkpoints=["a.pt", "b.pt"],
    )

    params = load_decision_params(path, expected_class_names=["rain", "sunny"])

    assert params.checkpoints == ["a.pt", "b.pt"]
