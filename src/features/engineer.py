"""
Stage 3a — Feature engineering.

Every transform here traces back to a specific finding in the Stage-2 EDA:

  EDA finding                          -> feature built here
  ------------------------------------------------------------------
  lead_time right-skewed (skew 1.42)   -> log1p transform + buckets
  zero-inflated counts                 -> binary "has any" flags
  lead_time x deposit_type interaction -> explicit interaction feature
  strong monthly seasonality           -> cyclical sin/cos month encoding
  risk profile (long-lead + group +    -> composite risk score
    non-refundable + few requests)
  guest composition matters            -> total_guests, has_children

DESIGN RULE: everything in this module is ROW-WISE and STATELESS — each feature
depends only on values within the same row, never on dataset-level statistics.
That makes it safe to apply to train, test, and live production rows identically,
with zero leakage risk. Anything that must LEARN from data (target encoding,
scaling) lives in encode.py / Stage 4, fit on train only.

Run:  python src/features/engineer.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("engineer")

TRAIN_IN = "data/processed/train.csv"
TEST_IN = "data/processed/test.csv"
TRAIN_OUT = "data/processed/train_featured.csv"
TEST_OUT = "data/processed/test_featured.csv"

MONTH_NUM = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

# Lead-time buckets chosen from the EDA distribution, not arbitrary round numbers:
# same-week, within-month, quarter, half-year, and the long tail.
LEAD_BINS = [-1, 7, 30, 90, 180, 10_000]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "180+"]


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Skew corrections. log1p (not log) because lead_time legitimately hits 0."""
    df = df.copy()
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["adr_log"] = np.log1p(df["adr"].clip(lower=0))
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-inflated counts carry most of their signal in has/hasn't, not in the
    magnitude. Keep the raw count too and let the model decide."""
    df = df.copy()
    df["has_prev_cancellation"] = (df["previous_cancellations"] > 0).astype(int)
    df["has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
    df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)
    df["has_children"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    return df


def add_guest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Party composition. total_guests is a better size signal than adults alone."""
    df = df.copy()
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_solo"] = (df["total_guests"] == 1).astype(int)
    df["is_family"] = ((df["adults"] >= 1) & ((df["children"] + df["babies"]) > 0)).astype(int)
    # Price per guest: the same ADR means something different for 1 vs 4 people.
    df["adr_per_guest"] = df["adr"] / df["total_guests"].clip(lower=1)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding for month.

    Why sin/cos rather than an integer 1..12: December (12) and January (1) are
    adjacent in time but maximally distant as integers. Projecting onto a circle
    makes that adjacency explicit, so a model can learn 'winter' rather than
    'high month number'.
    """
    df = df.copy()
    m = df["arrival_date_month"].map(MONTH_NUM)
    df["arrival_month_num"] = m
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["is_summer"] = m.isin([6, 7, 8]).astype(int)
    df["is_peak_season"] = m.isin([6, 7, 8, 12]).astype(int)
    return df


def add_lead_time_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Discretising lead_time lets linear models capture its non-linear effect on
    cancellation without needing a spline."""
    df = df.copy()
    df["lead_time_bucket"] = pd.cut(df["lead_time"], bins=LEAD_BINS, labels=LEAD_LABELS).astype(str)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """The EDA heatmap showed cancellation rate climbing with lead time at a
    different rate per deposit type — a genuine interaction. Trees find these on
    their own, but making them explicit helps linear/regularised models and makes
    the effect interpretable."""
    df = df.copy()
    df["lead_x_nonrefund"] = df["lead_time_log"] * (df["deposit_type"] == "Non Refund").astype(int)
    df["lead_x_groups"] = df["lead_time_log"] * (df["market_segment"] == "Groups").astype(int)
    df["requests_x_repeat"] = df["total_of_special_requests"] * df["is_repeated_guest"]
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """A composite, hand-built risk score straight from the EDA segment profile.

    This is deliberately simple and interpretable — it is NOT a model. Its value
    is (a) as a strong single baseline feature, and (b) as something a business
    stakeholder can read and sanity-check. Weights reflect the direction and rough
    magnitude of each effect seen in the EDA, not fitted coefficients.
    """
    df = df.copy()
    score = (
        2.0 * (df["deposit_type"] == "Non Refund").astype(int)
        + 1.5 * (df["market_segment"] == "Groups").astype(int)
        + 1.0 * (df["lead_time"] > 180).astype(int)
        + 1.0 * df["has_prev_cancellation"]
        - 1.0 * df["has_special_requests"]
        - 1.0 * df["is_repeated_guest"]
    )
    df["risk_score"] = score
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full row-wise feature pipeline. Order matters: transforms and flags feed
    the interaction and risk-score steps."""
    n_before = df.shape[1]
    df = add_transforms(df)
    df = add_binary_flags(df)
    df = add_guest_features(df)
    df = add_temporal_features(df)
    df = add_lead_time_buckets(df)
    df = add_interactions(df)
    df = add_risk_score(df)
    log.info("Engineered %d -> %d columns (+%d)", n_before, df.shape[1], df.shape[1] - n_before)
    return df


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_IN)
    test = pd.read_csv(TEST_IN)
    log.info("Loaded train=%d, test=%d", len(train), len(test))

    # Safe to apply to both: every transform is row-wise, learns nothing.
    train_f = engineer(train)
    test_f = engineer(test)

    train_f.to_csv(TRAIN_OUT, index=False)
    test_f.to_csv(TEST_OUT, index=False)
    log.info("Wrote %s and %s", TRAIN_OUT, TEST_OUT)

    new_cols = [c for c in train_f.columns if c not in train.columns]
    log.info("New features (%d): %s", len(new_cols), new_cols)
