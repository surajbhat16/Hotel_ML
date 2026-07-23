"""
Stage 2b — Exploratory Data Analysis (TRAIN ONLY) + leakage hunting + the
correct, leakage-free imputation of `country`.

Philosophy: every analysis answers a specific QUESTION and drives a DECISION.
We don't produce 40 plots and shrug. Each function returns a small, printable
finding so the pipeline is reproducible and the insights are logged.

CRITICAL: this module reads ONLY train.csv. The test set stays sealed until final
evaluation. Any value learned here (e.g. the modal country) is learned on train
and later applied to test.

Run:  python src/processing/eda.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eda")

TRAIN = "data/processed/train.csv"
TARGET = "is_canceled"


def class_balance(df: pd.DataFrame) -> float:
    """Q: how imbalanced is the target? Decision: drives metric choice (PR-AUC
    over accuracy) and imbalance handling (class weights / SMOTE) downstream."""
    rate = float(df[TARGET].mean())
    log.info(
        "Class balance — cancellations: %.1f%% (imbalance ratio ~%.1f:1)",
        100 * rate,
        (1 - rate) / rate,
    )
    return rate


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Q: what do the key numeric distributions look like (skew, spread)?
    Decision: skewed features (lead_time, adr) may need log-transform or robust
    scaling in Stage 4."""
    cols = ["lead_time", "adr", "total_of_special_requests", "booking_changes"]
    summary = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    summary["skew"] = df[cols].skew()
    log.info("Numeric summary:\n%s", summary.round(2).to_string())
    return summary


def cancellation_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Q: does cancellation rate differ across a categorical's levels?
    Decision: strong differences => the feature is predictive => keep/encode it.
    Returns a rate table sorted by cancellation rate."""
    tab = (
        df.groupby(col)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "cancel_rate", "count": "n"})
        .sort_values("cancel_rate", ascending=False)
    )
    log.info("Cancellation rate by %s:\n%s", col, tab.round(3).to_string())
    return tab


def leakage_scan(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Systematic leakage hunt. For each numeric feature, compute correlation
    with the target; for categoricals, the max class cancellation rate. A single
    feature that 'explains' almost everything is almost always leakage, not gold.

    Decision: anything above threshold gets manually interrogated — does this
    value exist at prediction time? If not, drop it.
    """
    num = df.select_dtypes(include=[np.number]).drop(columns=[TARGET])
    corrs = num.corrwith(df[TARGET]).abs().sort_values(ascending=False)
    log.info("Top |correlation| with target:\n%s", corrs.head(8).round(3).to_string())

    suspicious = corrs[corrs > threshold]
    if len(suspicious):
        log.warning("POTENTIAL LEAKAGE (|corr| > %.2f): %s", threshold, dict(suspicious.round(3)))
    else:
        log.info(
            "No single numeric feature exceeds |corr| %.2f — no obvious numeric leakage.", threshold
        )
    return corrs


def missingness_report(df: pd.DataFrame) -> pd.Series:
    """Q: what's still missing, and does missingness correlate with the target?
    Decision: confirms whether missingness flags carry signal."""
    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        log.info("Remaining missingness (fraction):\n%s", miss.round(4).to_string())
    else:
        log.info("No missing values remain.")
    return miss


def impute_country_from_train(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The correct, leakage-free imputation we deferred in Stage 1.

    The fill value (modal country) is learned from TRAIN ONLY, then applied to
    both splits. Computing it on the full data would leak test into train.
    """
    fill = train["country"].mode(dropna=True).iloc[0]
    train = train.copy()
    test = test.copy()
    # Preserve the 'was missing' signal before filling.
    for part in (train, test):
        part["country_missing"] = part["country"].isna().astype(int)
    train["country"] = train["country"].fillna(fill)
    test["country"] = test["country"].fillna(fill)
    log.info("Imputed country with TRAIN mode '%s' (+ country_missing flag)", fill)
    return train, test, fill


if __name__ == "__main__":
    train = pd.read_csv(TRAIN)
    log.info("Loaded train: %d rows (test stays sealed)", len(train))

    log.info("=" * 60)
    class_balance(train)
    log.info("=" * 60)
    numeric_summary(train)
    log.info("=" * 60)
    for col in ["deposit_type", "market_segment", "customer_type"]:
        cancellation_by_category(train, col)
        log.info("-" * 40)
    log.info("=" * 60)
    leakage_scan(train)
    log.info("=" * 60)
    missingness_report(train)
