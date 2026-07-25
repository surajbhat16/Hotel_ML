"""Tests for Stage 6 metrics + training. Run: pytest tests/ -q

Training is slow, so these tests use tiny synthetic data, few Optuna trials, and
assert on structure/contracts rather than exact scores.
"""

import numpy as np
import pandas as pd

from src.models import metrics, train


def _toy(n: int = 400) -> pd.DataFrame:
    """Small frame with genuine signal so models beat the dummy, plus the
    arrival_month_num column the time-series CV needs."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, n)
    y = (signal + rng.normal(0, 0.4, n) > 0).astype(int)
    return pd.DataFrame(
        {
            "is_canceled": y,
            "arrival_month_num": np.sort(rng.integers(1, 13, n)),
            "strong": signal,
            "noise": rng.normal(0, 1, n),
        }
    )


# --------------------------------- metrics ---------------------------------


def test_evaluate_returns_all_metrics():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.3, 0.8, 0.6, 0.2])
    m = metrics.evaluate(y, p)
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier"]:
        assert key in m
        assert 0.0 <= m[key] <= 1.0


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = metrics.evaluate(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_best_threshold_improves_or_matches_f1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.3, 200), 0, 1)
    thr, thr_f1 = metrics.best_threshold_for_f1(y, p)
    default_f1 = metrics.evaluate(y, p, threshold=0.5)["f1"]
    assert thr_f1 >= default_f1 - 1e-9
    assert 0.05 <= thr <= 0.95


# --------------------------------- training --------------------------------


def test_cv_score_returns_valid_prauc():
    df = _toy()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    from sklearn.linear_model import LogisticRegression

    score = train.cv_score(lambda: LogisticRegression(max_iter=500), x, y, n_splits=3)
    assert 0.0 <= score <= 1.0


def test_model_beats_dummy_on_signal():
    """A real model should beat the prior-only dummy when signal exists."""
    df = _toy()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    dummy = train.cv_score(lambda: DummyClassifier(strategy="prior"), x, y, n_splits=3)
    logreg = train.cv_score(lambda: LogisticRegression(max_iter=500), x, y, n_splits=3)
    assert logreg > dummy


def test_cv_uses_time_order_no_future_leak():
    """cv_score must consume the time-series splits — verify the adapter yields
    folds whose validation month is >= training month."""
    df = _toy()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    order = x["arrival_month_num"].to_numpy()
    for tr, va in train.expanding_window_splits_from_xy(x, y, n_splits=3):
        assert order[tr].max() <= order[va].min()


def test_tune_returns_best_params():
    df = _toy()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    result = train.tune(x, y, n_trials=3)
    assert "best_params" in result and "best_value" in result
    assert "n_estimators" in result["best_params"]
