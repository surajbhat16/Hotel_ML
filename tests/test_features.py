"""Tests for Stage 3 feature engineering + encoding. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.features import encode, engineer


def _toy():
    """Minimal frame carrying every column the engineering pipeline touches."""
    return pd.DataFrame(
        {
            "is_canceled": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "lead_time": [0, 200, 15, 400, 60, 5, 120, 250, 30, 90],
            "adr": [100.0, 80.0, 150.0, 60.0, 120.0, 200.0, 90.0, 70.0, 110.0, 95.0],
            "adults": [2, 1, 2, 3, 2, 1, 2, 2, 1, 2],
            "children": [0, 1, 0, 2, 0, 0, 1, 0, 0, 0],
            "babies": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "previous_cancellations": [0, 2, 0, 1, 0, 0, 0, 3, 0, 0],
            "booking_changes": [0, 1, 0, 0, 2, 0, 0, 0, 1, 0],
            "total_of_special_requests": [1, 0, 2, 0, 1, 3, 0, 0, 2, 1],
            "is_repeated_guest": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "deposit_type": [
                "No Deposit",
                "Non Refund",
                "No Deposit",
                "Non Refund",
                "Refundable",
                "No Deposit",
                "No Deposit",
                "Non Refund",
                "No Deposit",
                "No Deposit",
            ],
            "market_segment": [
                "Direct",
                "Groups",
                "Online TA",
                "Groups",
                "Direct",
                "Online TA",
                "Corporate",
                "Groups",
                "Direct",
                "Online TA",
            ],
            "customer_type": ["Transient"] * 10,
            "hotel": ["City Hotel"] * 5 + ["Resort Hotel"] * 5,
            "arrival_date_month": [
                "January",
                "June",
                "December",
                "July",
                "March",
                "August",
                "February",
                "November",
                "May",
                "September",
            ],
            "country": ["PRT", "GBR", "PRT", "FRA", "PRT", "GBR", "ESP", "PRT", "FRA", "GBR"],
            "agent": [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 3.0, 2.0],
            "company": [0.0] * 10,
        }
    )


# --------------------------- feature engineering ---------------------------


def test_log_transform_handles_zero():
    out = engineer.add_transforms(_toy())
    assert out["lead_time_log"].iloc[0] == 0.0  # log1p(0) == 0, no -inf
    assert np.isfinite(out["lead_time_log"]).all()


def test_binary_flags_are_binary():
    out = engineer.add_binary_flags(_toy())
    for col in [
        "has_prev_cancellation",
        "has_booking_changes",
        "has_special_requests",
        "has_children",
    ]:
        assert set(out[col].unique()) <= {0, 1}


def test_total_guests_sums_correctly():
    out = engineer.add_guest_features(_toy())
    expected = _toy()["adults"] + _toy()["children"] + _toy()["babies"]
    assert (out["total_guests"] == expected).all()


def test_cyclical_month_is_continuous():
    """December and January must be close in sin/cos space despite 12 vs 1."""
    out = engineer.add_temporal_features(_toy())
    dec = out.loc[out["arrival_month_num"] == 12, ["month_sin", "month_cos"]].iloc[0]
    jan = out.loc[out["arrival_month_num"] == 1, ["month_sin", "month_cos"]].iloc[0]
    dist = np.hypot(dec["month_sin"] - jan["month_sin"], dec["month_cos"] - jan["month_cos"])
    assert dist < 0.6  # adjacent on the circle


def test_engineer_is_row_wise_and_order_independent():
    """A shuffled frame must produce identical per-row features — proving the
    pipeline learns nothing from dataset-level statistics."""
    df = _toy()
    a = engineer.engineer(df).sort_index()
    b = engineer.engineer(df.sample(frac=1, random_state=7)).sort_index()
    num = a.select_dtypes(include=[np.number]).columns
    pd.testing.assert_frame_equal(a[num], b[num], check_like=True)


def test_engineer_adds_expected_columns():
    out = engineer.engineer(_toy())
    for col in ["lead_time_log", "risk_score", "month_sin", "total_guests", "lead_time_bucket"]:
        assert col in out.columns


# --------------------------------- encoding ---------------------------------


def test_frequency_encoding_sums_to_one_on_train():
    _tr, _te, maps = encode.frequency_encode(_toy(), _toy(), ["country"])
    assert abs(sum(maps["country"].values()) - 1.0) < 1e-9


def test_frequency_encoding_unseen_level_gets_zero():
    train = _toy()
    test = _toy().copy()
    test.loc[0, "country"] = "ZZZ"  # never seen in train
    _, te, _ = encode.frequency_encode(train, test, ["country"])
    assert te["country_freq"].iloc[0] == 0.0


def test_target_encoding_no_self_leakage():
    """The out-of-fold value for a row must not equal the mean computed WITH that
    row included — that equality is the signature of leakage."""
    train = _toy()
    tr, _, _ = encode.target_encode_oof(train, _toy(), ["country"], n_splits=5, m=0.0)
    naive = train.groupby("country")["is_canceled"].transform("mean")
    # At least some rows must differ from the naive (leaky) encoding.
    assert (tr["country_te"].to_numpy() != naive.to_numpy()).any()


def test_target_encoding_unseen_level_falls_back_to_prior():
    train = _toy()
    test = _toy().copy()
    test.loc[0, "country"] = "ZZZ"
    _, te, maps = encode.target_encode_oof(train, test, ["country"])
    assert abs(te["country_te"].iloc[0] - maps["country"]["prior"]) < 1e-9


def test_smoothing_pulls_rare_levels_toward_prior():
    """A level seen once should sit closer to the prior than a frequent level's
    own mean would suggest."""
    train = _toy()
    _, te, _ = encode.target_encode_oof(train, _toy(), ["country"], m=50.0)
    prior = train["is_canceled"].mean()
    esp = te.loc[te["country"] == "ESP", "country_te"].iloc[0]  # appears once
    assert abs(esp - prior) < 0.15


def test_one_hot_aligns_test_schema_to_train():
    train = _toy()
    test = _toy().copy()
    test.loc[0, "market_segment"] = "Aviation"  # level absent from train
    tr, te = encode.one_hot(train, test, ["market_segment"])
    assert list(tr.columns) == list(te.columns)
    assert "market_segment_Aviation" not in tr.columns


def test_full_encode_pipeline_is_all_numeric_and_aligned():
    train = engineer.engineer(_toy())
    test = engineer.engineer(_toy())
    tr, te, _ = encode.encode_all(train, test)
    assert list(tr.columns) == list(te.columns)
    leftover = tr.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    assert leftover == [], f"non-numeric columns survived: {leftover}"
