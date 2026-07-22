"""
Stage 1 — Data cleaning for the hotel-booking dataset.

Design principles (the things interviewers probe):
  - RAW DATA IS IMMUTABLE. We read from data/raw and write to data/interim.
    Every transformation is reproducible from raw by rerunning this file.
  - CLEANING DECISIONS ARE LOGGED, not silent. A silent .dropna() that removes
    30% of rows is how you ship a broken model. We count and report every drop.
  - FIT-ON-TRAIN-ONLY discipline starts here in spirit: any imputation *value*
    (a median, a mode) must ultimately be learned on the training split, never
    the whole dataset, or you leak. In this module we separate "structural"
    cleaning (safe on all rows: dedup, fixing impossible values) from
    "statistical" imputation (must be fit on train). See the docstrings.

Run:  python src/processing/clean.py
"""

from __future__ import annotations

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("clean")

RAW = "data/raw/hotel_bookings.csv"
OUT = "data/interim/hotel_clean.csv"

# The leakage column: reservation_status is recorded AFTER we know the outcome,
# so it perfectly encodes is_canceled. Keeping it = a model that "predicts" the
# past. We drop it here so it can never sneak into features downstream.
LEAKAGE_COLS = ["reservation_status"]


def load_raw(path: str = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded raw: %d rows, %d cols", len(df), df.shape[1])
    return df


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LEAKAGE_COLS if c in df.columns]
    if present:
        log.info("Dropping leakage columns: %s", present)
    return df.drop(columns=present)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Exact-duplicate rows are almost always double-ingested records. Safe to
    drop on all rows (structural, no statistics learned)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info(
        "Removed %d exact duplicate rows (%.2f%%)",
        before - len(df),
        100 * (before - len(df)) / before,
    )
    return df


def fix_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Structural fixes for values that violate domain rules. These are decided
    by business logic, not by statistics, so they're safe to apply to all rows.

      - Negative ADR is impossible (you can't be paid to stay). Clip to 0.
      - The famous ADR=5400 outlier is a data-entry error; cap at a sane ceiling.
      - Zero-guest bookings (adults+children+babies == 0) are invalid records.
    """
    df = df.copy()

    neg_adr = (df["adr"] < 0).sum()
    df.loc[df["adr"] < 0, "adr"] = 0.0
    log.info("Clipped %d negative ADR values to 0", neg_adr)

    # Cap the extreme outlier. We use a domain ceiling (1000/night) rather than a
    # purely statistical one so the rule is stable across future data batches.
    ceiling = 1000.0
    capped = (df["adr"] > ceiling).sum()
    df.loc[df["adr"] > ceiling, "adr"] = ceiling
    log.info("Capped %d ADR values above %.0f", capped, ceiling)

    guests = df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    zero_guest = (guests == 0).sum()
    df = df[guests > 0].reset_index(drop=True)
    log.info("Dropped %d zero-guest bookings", zero_guest)

    return df


def handle_missing_structural(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness that has a *meaning*, handled without learning statistics.

    - agent / company: NaN means 'no agent / no company involved', not
      'unknown'. The correct fill is a sentinel (0), and this is domain-driven,
      so it's safe on all rows. We also add explicit missingness flags because
      'was there an agent at all' is itself predictive.
    - children: only a handful missing; fill with 0 (the mode by far) — but we
      flag that in a real pipeline this tiny imputation is done on train only.
    """
    df = df.copy()

    for col in ["agent", "company"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)
        log.info("Filled %s NaN with sentinel 0 (+ missingness flag)", col)

    # children: 4 missing. Mode is 0. (In the real pipeline the fill value is the
    # TRAIN median/mode — see impute_statistical for the correct discipline.)
    df["children"] = df["children"].fillna(0).astype(int)

    return df


def impute_statistical(
    train: pd.DataFrame, other: pd.DataFrame, col: str = "country"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The CORRECT way to do statistical imputation without leakage.

    The fill value (here, the modal country) is learned from TRAIN ONLY, then
    applied to both train and any held-out split. If you compute the mode on the
    full dataset you leak information from validation/test into training.

    Returns (train_filled, other_filled).
    """
    fill_value = train[col].mode(dropna=True).iloc[0]
    log.info("Imputing '%s' with train-learned mode: %s", col, fill_value)
    train = train.copy()
    other = other.copy()
    train[col] = train[col].fillna(fill_value)
    other[col] = other[col].fillna(fill_value)
    return train, other


def clean_pipeline(path: str = RAW) -> pd.DataFrame:
    """Structural cleaning only (safe on all rows). Statistical imputation of
    'country' is demonstrated separately in impute_statistical, which must run
    AFTER the train/test split in Stage 2."""
    df = load_raw(path)
    df = drop_leakage(df)
    df = remove_duplicates(df)
    df = fix_impossible_values(df)
    df = handle_missing_structural(df)
    remaining_na = df.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if len(remaining_na):
        log.info("Columns still holding NaN (to impute post-split): %s", dict(remaining_na))
    log.info("Clean (structural) done: %d rows, %d cols", len(df), df.shape[1])
    return df


if __name__ == "__main__":
    cleaned = clean_pipeline()
    cleaned.to_csv(OUT, index=False)
    log.info("Wrote %s", OUT)
