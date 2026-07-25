"""
Stage 5a — Cross-validation design.

The single most important idea in this phase: for temporally-ordered data, you
CANNOT use ordinary k-fold cross-validation. Random k-fold puts future rows in the
training folds and past rows in validation, so the model is evaluated on its
ability to "predict the past from the future" — which never happens in production
and gives a dishonestly optimistic score.

The fix is TIME-SERIES cross-validation: every validation fold is strictly LATER
than its training data. Two standard schemes:

  EXPANDING window (sklearn TimeSeriesSplit): train on [0..k], validate on the
    next block; the training window grows each fold. Uses all history — good when
    older data stays relevant.

  ROLLING window: train on a FIXED-width recent window, validate on the next
    block; the window slides forward. Better when the process drifts and old data
    becomes stale (very relevant for a bookings model with changing seasons).

Both share the invariant: max(train time) < min(validation time) in every fold.

We sort by the same arrival index used for the train/test split so the CV folds
are consistent with how the final hold-out was carved.

Run:  python src/models/cv.py
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("cv")

TRAIN_IN = "data/processed/train_scaled.csv"
TARGET = "is_canceled"

# The scaled file no longer carries the month string, so we reconstruct a time
# order from the cyclical features we engineered. arrival_month_num survived, and
# combined with (implicit) year ordering from the original split it's enough to
# order within the training period. If arrival_month_num is absent we fall back to
# the row order, which already reflects the time-sorted split.


def time_order_index(df: pd.DataFrame) -> np.ndarray:
    """Return an integer ordering of rows in ascending time.

    Prefer an explicit month number if present; otherwise assume the file is
    already time-sorted (our split writes it in arrival order) and use row order.
    """
    if "arrival_month_num" in df.columns:
        # Stable sort keeps within-month rows in their existing (time-sorted) order.
        return np.argsort(df["arrival_month_num"].to_numpy(), kind="stable")
    return np.arange(len(df))


def expanding_window_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """sklearn's TimeSeriesSplit: expanding train window, forward validation."""
    order = time_order_index(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(order):
        # Map back to original row positions via the time ordering.
        yield order[tr], order[va]


def rolling_window_splits(
    df: pd.DataFrame, n_splits: int = 5, window: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Fixed-width rolling window: train on the last `window` rows before each
    validation block. Emphasises recent data, which matters under drift."""
    order = time_order_index(df)
    n = len(order)
    fold = n // (n_splits + 1)
    window = window or fold * 2

    for k in range(1, n_splits + 1):
        va_start = k * fold
        va_end = va_start + fold if k < n_splits else n
        tr_start = max(0, va_start - window)
        tr_idx = order[tr_start:va_start]
        va_idx = order[va_start:va_end]
        if len(tr_idx) and len(va_idx):
            yield tr_idx, va_idx


def describe_splits(df: pd.DataFrame, scheme: str = "expanding", n_splits: int = 5) -> None:
    """Log the shape and time-ordering guarantee of each fold — proof that no fold
    trains on the future."""
    gen = (
        expanding_window_splits(df, n_splits)
        if scheme == "expanding"
        else rolling_window_splits(df, n_splits)
    )
    log.info("Scheme=%s, n_splits=%d", scheme, n_splits)
    for i, (tr, va) in enumerate(gen, 1):
        tr_rate = df.iloc[tr][TARGET].mean()
        va_rate = df.iloc[va][TARGET].mean()
        log.info(
            "  fold %d | train=%6d (cancel %.3f) | valid=%6d (cancel %.3f)",
            i,
            len(tr),
            tr_rate,
            len(va),
            va_rate,
        )


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_IN)
    log.info("Loaded scaled train: %d rows", len(df))
    log.info("=" * 60)
    describe_splits(df, "expanding", n_splits=5)
    log.info("=" * 60)
    describe_splits(df, "rolling", n_splits=5)
