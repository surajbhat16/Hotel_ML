"""Unit tests for Stage 1 cleaning. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.processing import clean


def _toy():
    return pd.DataFrame(
        {
            "adr": [100.0, -5.0, 5400.0, 50.0],
            "adults": [2, 1, 0, 2],
            "children": [0.0, np.nan, 0.0, 1.0],
            "babies": [0, 0, 0, 0],
            "agent": [1.0, np.nan, 3.0, np.nan],
            "company": [np.nan, np.nan, 10.0, np.nan],
            "country": ["PRT", np.nan, "GBR", "PRT"],
            "reservation_status": ["Check-Out", "Canceled", "Check-Out", "No-Show"],
        }
    )


def test_leakage_dropped():
    out = clean.drop_leakage(_toy())
    assert "reservation_status" not in out.columns


def test_negative_adr_clipped():
    out = clean.fix_impossible_values(_toy())
    assert (out["adr"] >= 0).all()


def test_extreme_adr_capped():
    out = clean.fix_impossible_values(_toy())
    assert out["adr"].max() <= 1000.0


def test_zero_guest_dropped():
    out = clean.fix_impossible_values(_toy())
    guests = out["adults"] + out["children"].fillna(0) + out["babies"]
    assert (guests > 0).all()


def test_agent_sentinel_and_flag():
    out = clean.handle_missing_structural(_toy())
    assert out["agent"].isna().sum() == 0
    assert "agent_missing" in out.columns
    assert out["agent_missing"].sum() == 2


def test_statistical_imputation_uses_train_mode_only():
    train = pd.DataFrame({"country": ["PRT", "PRT", np.nan]})
    other = pd.DataFrame({"country": [np.nan]})
    t, o = clean.impute_statistical(train, other, "country")
    assert t["country"].isna().sum() == 0
    assert o["country"].iloc[0] == "PRT"  # filled with TRAIN mode
