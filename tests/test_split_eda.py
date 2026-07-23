"""Tests for Stage 2 split + EDA. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.processing import eda, split


def _toy_time():
    return pd.DataFrame(
        {
            "arrival_date_year": [2015, 2015, 2016, 2016, 2017, 2017],
            "arrival_date_month": ["January", "June", "March", "August", "May", "December"],
            "is_canceled": [0, 1, 0, 1, 0, 1],
            "lead_time": [10, 200, 30, 150, 40, 300],
        }
    )


def test_arrival_index_orders_in_time():
    out = split.add_arrival_index(_toy_time())
    assert list(out["_arrival_index"]) == [201501, 201506, 201603, 201608, 201705, 201712]


def test_time_split_has_no_month_overlap():
    train, test = split.time_split(_toy_time(), test_size=0.34)
    train = split.add_arrival_index(train)
    test = split.add_arrival_index(test)
    # The latest arrivals must be in test, earliest in train — no overlap.
    assert train["_arrival_index"].max() < test["_arrival_index"].min()


def test_time_split_never_empty():
    train, test = split.time_split(_toy_time(), test_size=0.2)
    assert len(train) > 0 and len(test) > 0


def test_class_balance_returns_rate():
    rate = eda.class_balance(_toy_time())
    assert 0.0 <= rate <= 1.0


def test_country_imputation_uses_train_mode_and_flags():
    train = pd.DataFrame({"country": ["PRT", "PRT", np.nan, "GBR"]})
    test = pd.DataFrame({"country": [np.nan, "FRA"]})
    tr, te, fill = eda.impute_country_from_train(train, test)
    assert fill == "PRT"  # learned on train
    assert tr["country"].isna().sum() == 0
    assert te["country"].iloc[0] == "PRT"  # test filled with TRAIN mode
    assert "country_missing" in tr.columns
    assert tr["country_missing"].sum() == 1


def test_leakage_scan_flags_perfect_feature():
    df = _toy_time().copy()
    df["cheat"] = df["is_canceled"]  # perfect leak
    corrs = eda.leakage_scan(df, threshold=0.85)
    assert corrs["cheat"] > 0.85
