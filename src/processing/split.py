"""
Stage 2a — Train/test splitting.

The split is the FIRST thing that happens after structural cleaning, and it
happens BEFORE any EDA or statistical imputation. Everything the model learns —
distributions, imputation values, encodings, scalers — must come from TRAIN only.
Touching test before final evaluation is data snooping and inflates your score.

Two strategies provided:
  - time_split: the correct default for this dataset. Train on earlier arrivals,
    test on later ones, mirroring production (train on past, predict future).
    A random split would leak future seasonality into training.
  - stratified_split: random but preserves the class balance. Useful ONLY when
    rows are exchangeable (no time structure), kept here for comparison/teaching.

Run:  python src/processing/split.py
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("split")

INTERIM = "data/interim/hotel_clean.csv"
TRAIN_OUT = "data/processed/train.csv"
TEST_OUT = "data/processed/test.csv"
TARGET = "is_canceled"

MONTH_ORDER = {
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


def add_arrival_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single sortable integer for arrival (year*100 + month) so we can
    order bookings in time without a full date column."""
    df = df.copy()
    df["_arrival_index"] = df["arrival_date_year"] * 100 + df["arrival_date_month"].map(MONTH_ORDER)
    return df


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the latest `test_size` fraction of arrivals becomes test.

    We pick a cutoff on the sorted arrival index such that ~test_size of rows
    fall after it. Ties on the boundary month all go to the same side to avoid
    the same month appearing in both splits.
    """
    df = add_arrival_index(df)
    cutoff = df["_arrival_index"].quantile(1 - test_size)
    train = df[df["_arrival_index"] <= cutoff].copy()
    test = df[df["_arrival_index"] > cutoff].copy()

    # Guard against degenerate split if one period dominates.
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Time split produced an empty side; check arrival span.")

    for part in (train, test):
        part.drop(columns="_arrival_index", inplace=True)

    log.info(
        "Time split @ cutoff %d: train=%d (%.1f%%), test=%d (%.1f%%)",
        int(cutoff),
        len(train),
        100 * len(train) / len(df),
        len(test),
        100 * len(test) / len(df),
    )
    log.info(
        "Cancellation rate — train: %.3f, test: %.3f",
        train[TARGET].mean(),
        test[TARGET].mean(),
    )
    return train, test


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split that preserves class balance. For comparison only — NOT the
    right choice for temporally structured data like this."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    log.info(
        "Stratified split: train=%d, test=%d | cancel train %.3f / test %.3f",
        len(train),
        len(test),
        train[TARGET].mean(),
        test[TARGET].mean(),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    import os

    clean = pd.read_csv(INTERIM)
    log.info("Loaded clean interim: %d rows", len(clean))

    train, test = time_split(clean, test_size=0.2)

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    log.info("Wrote %s and %s", TRAIN_OUT, TEST_OUT)
