"""Tests for Stage 7 evaluation + explainability. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.models import evaluate, explain

# --------------------------------- evaluation ---------------------------------


def test_roc_points_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = evaluate.roc_points(y, p)
    assert r["auc"] > 0.99
    assert len(r["fpr"]) == len(r["tpr"])


def test_pr_points_reports_baseline():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9])
    r = evaluate.pr_points(y, p)
    assert abs(r["baseline"] - 0.25) < 1e-9


def test_calibration_report_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    r = evaluate.calibration_report(y, p, n_bins=5)
    assert {"brier", "ece", "mean_predicted", "fraction_positive"} <= set(r)
    assert 0.0 <= r["brier"] <= 1.0
    assert 0.0 <= r["ece"] <= 1.0


def test_isotonic_calibration_improves_miscalibrated_probs():
    """Deliberately squash probabilities toward 0.5 (under-confident), then check
    isotonic calibration reduces Brier on held-out data."""
    rng = np.random.default_rng(1)
    n = 2000
    true_p = rng.random(n)
    y = (rng.random(n) < true_p).astype(int)
    # Miscalibrate: compress toward 0.5.
    bad_p = 0.5 + (true_p - 0.5) * 0.3

    # Split into calibration and test halves.
    cal, test = slice(0, n // 2), slice(n // 2, n)
    before = evaluate.brier_score_loss(y[test], bad_p[test])
    fixed = evaluate.calibrate_isotonic(y[cal], bad_p[cal], bad_p[test])
    after = evaluate.brier_score_loss(y[test], fixed)
    assert after <= before + 1e-6


def test_cost_curve_prefers_low_threshold_when_fn_expensive():
    """If false negatives are very costly, the optimal threshold should be low
    (catch more positives)."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    r = evaluate.cost_curve(y, p, fn_cost=20.0, fp_cost=1.0)
    assert r["best_threshold"] <= 0.5


# -------------------------------- explainability ------------------------------


def _tiny_model():
    """Train a tiny LightGBM booster on synthetic data with clear signal."""
    import lightgbm as lgb

    rng = np.random.default_rng(3)
    n = 400
    strong = rng.normal(0, 1, n)
    y = (strong + rng.normal(0, 0.3, n) > 0).astype(int)
    x = pd.DataFrame({"strong": strong, "noise": rng.normal(0, 1, n)})
    ds = lgb.Dataset(x, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 8},
        ds,
        num_boost_round=30,
    )
    return booster, x


def test_shap_global_importance_ranks_signal_first():
    booster, x = _tiny_model()
    shap_values, x_s = explain.compute_shap(booster, x, sample=200)
    imp = explain.global_importance(shap_values, x_s)
    assert imp.index[0] == "strong"


def test_shap_values_shape_matches_data():
    booster, x = _tiny_model()
    shap_values, x_s = explain.compute_shap(booster, x, sample=150)
    assert shap_values.shape == x_s.shape


def test_signed_effect_detects_direction():
    booster, x = _tiny_model()
    shap_values, x_s = explain.compute_shap(booster, x, sample=200)
    signed = explain.signed_effect(shap_values, x_s, top=2)
    strong_row = signed[signed["feature"] == "strong"].iloc[0]
    # 'strong' positively drives the positive class -> ↑ direction.
    assert strong_row["direction"] == "↑cancel"


def test_local_explanation_structure():
    booster, x = _tiny_model()
    shap_values, x_s = explain.compute_shap(booster, x, sample=100)
    ex = explain.local_explanation(booster, shap_values, x_s, row=0)
    assert {"base_value", "prediction_logit", "top_contributions"} <= set(ex)
    assert len(ex["top_contributions"]) >= 1
