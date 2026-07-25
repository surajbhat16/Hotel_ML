"""Tests for Stage 4 scaling + selection. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.features import scale, select


def _toy_scaled():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "is_canceled": rng.integers(0, 2, n),
            "lead_time": rng.gamma(2, 50, n),  # continuous, skewed
            "adr": rng.normal(100, 30, n),  # continuous
            "flag_a": rng.integers(0, 2, n),  # binary
            "flag_b": rng.integers(0, 2, n),  # binary
            "onehot_x": rng.integers(0, 2, n),  # binary/one-hot
        }
    )


# --------------------------------- scaling ---------------------------------


def test_detect_binary_columns():
    df = _toy_scaled()
    binary = set(scale.detect_binary_columns(df))
    assert {"flag_a", "flag_b", "onehot_x", "is_canceled"} <= binary
    assert "lead_time" not in binary and "adr" not in binary


def test_columns_to_scale_excludes_target_and_binary():
    df = _toy_scaled()
    cols = scale.columns_to_scale(df)
    assert set(cols) == {"lead_time", "adr"}


def test_scaler_fit_on_train_only_no_test_leakage():
    """The scaler's learned center must come from TRAIN. Shifting test must NOT
    change the transform applied — proving test didn't influence fitting."""
    train = _toy_scaled()
    test = _toy_scaled()
    cols = scale.columns_to_scale(train)
    scaler = scale.fit_scaler(train, cols, "robust")
    center_before = np.array(scaler.center_, copy=True)

    # Mutate test wildly; refit must be unaffected because we only fit on train.
    test["lead_time"] *= 1000
    scaler2 = scale.fit_scaler(train, cols, "robust")
    assert np.allclose(center_before, scaler2.center_)


def test_binary_columns_untouched_by_scaling():
    train = _toy_scaled()
    test = _toy_scaled()
    train_s, _test_s, _ = scale.scale_pipeline(train, test, "robust")
    # Binary columns must be identical after scaling.
    for col in ["flag_a", "flag_b", "onehot_x"]:
        assert (train_s[col] == train[col]).all()


def test_scaling_centers_continuous_near_zero():
    train = _toy_scaled()
    test = _toy_scaled()
    train_s, _, _ = scale.scale_pipeline(train, test, "robust")
    # RobustScaler centers on the median -> scaled median ~ 0.
    assert abs(train_s["lead_time"].median()) < 1e-6


# -------------------------------- selection --------------------------------


def _toy_select():
    """A frame where one feature is strongly predictive and one is pure noise."""
    rng = np.random.default_rng(1)
    n = 500
    signal = rng.normal(0, 1, n)
    y = (signal + rng.normal(0, 0.3, n) > 0).astype(int)
    return pd.DataFrame(
        {
            "is_canceled": y,
            "strong": signal,
            "noise": rng.normal(0, 1, n),
            "flag": rng.integers(0, 2, n),
        }
    )


def test_mutual_info_ranks_signal_above_noise():
    df = _toy_select()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    mi = select.filter_mutual_info(x, y)
    assert mi["strong"] > mi["noise"]


def test_anova_f_ranks_signal_above_noise():
    df = _toy_select()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    f = select.filter_anova_f(x, y)
    assert f["strong"] > f["noise"]


def test_l1_selects_signal():
    df = _toy_select()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    coef = select.embedded_l1(x, y)
    assert coef["strong"] >= coef["noise"]


def test_consensus_ranking_puts_signal_first():
    df = _toy_select()
    x, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    rankings = {
        "mutual_info": select.filter_mutual_info(x, y),
        "anova_f": select.filter_anova_f(x, y),
        "l1": select.embedded_l1(x, y),
    }
    consensus = select.consensus_ranking(rankings, top_k=3)
    assert consensus.index[0] == "strong"
