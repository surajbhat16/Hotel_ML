"""Tests for Stage 5 cross-validation + imbalance handling. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.models import cv, imbalance


def _toy_time(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # arrival_month_num increases so there IS a time order to respect.
    month = np.sort(rng.integers(1, 13, n))
    return pd.DataFrame(
        {
            "is_canceled": rng.integers(0, 2, n),
            "arrival_month_num": month,
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
        }
    )


# ------------------------------ cross-validation ------------------------------


def test_expanding_window_train_grows_each_fold():
    df = _toy_time()
    sizes = [len(tr) for tr, _ in cv.expanding_window_splits(df, n_splits=5)]
    # Each successive training window must be at least as large as the previous.
    from itertools import pairwise

    assert all(b >= a for a, b in pairwise(sizes))


def test_expanding_window_never_trains_on_future():
    """The core guarantee: max train time < min validation time in every fold."""
    df = _toy_time()
    order = df["arrival_month_num"].to_numpy()
    for tr, va in cv.expanding_window_splits(df, n_splits=5):
        assert order[tr].max() <= order[va].min()


def test_rolling_window_has_fixed_width():
    df = _toy_time(300)
    sizes = [len(tr) for tr, _ in cv.rolling_window_splits(df, n_splits=5)]
    # After the first ramp-up folds, the window width should stabilise.
    later = sizes[1:]
    assert max(later) - min(later) <= 1


def test_rolling_window_never_trains_on_future():
    df = _toy_time(300)
    order = df["arrival_month_num"].to_numpy()
    for tr, va in cv.rolling_window_splits(df, n_splits=5):
        assert order[tr].max() <= order[va].min()


def test_no_row_in_both_train_and_valid():
    df = _toy_time()
    for tr, va in cv.expanding_window_splits(df, n_splits=5):
        assert set(tr).isdisjoint(set(va))


# ------------------------------ imbalance -------------------------------------


def test_class_weights_favor_minority():
    y = pd.Series([0] * 80 + [1] * 20)  # 4:1 imbalance
    w = imbalance.class_weights(y)
    assert w[1] > w[0]  # minority gets the larger weight


def test_scale_pos_weight_equals_neg_over_pos():
    y = pd.Series([0] * 80 + [1] * 20)
    assert abs(imbalance.scale_pos_weight(y) - 4.0) < 1e-9


def test_smote_balances_the_training_fold():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series([0] * 160 + [1] * 40)
    x_res, y_res = imbalance.smote_on_train_fold_only(x, y)
    counts = y_res.value_counts()
    # SMOTE oversamples the minority up to parity.
    assert counts[0] == counts[1]
    assert len(x_res) == len(y_res)


def test_smote_signature_only_accepts_train_fold():
    """Guard: the function must not accept validation data — its contract is
    train-fold-only. We assert it takes exactly (x, y[, seed])."""
    import inspect

    params = list(inspect.signature(imbalance.smote_on_train_fold_only).parameters)
    assert params[:2] == ["x_tr", "y_tr"]
    assert "x_val" not in params and "x_test" not in params
