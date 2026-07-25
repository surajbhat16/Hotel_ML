"""
Stage 3b — Categorical encoding (the leakage-critical part).

Three strategies, each matched to the cardinality and nature of the column:

  1. ONE-HOT for low-cardinality nominal columns (hotel, deposit_type,
     customer_type, market_segment, lead_time_bucket). Cheap, lossless, no
     ordinality implied. Cardinality is small so dimensionality stays sane.

  2. FREQUENCY encoding for high-cardinality columns (country, agent, company).
     Replaces a level with how often it occurs. Stateless-ish, robust, and does
     not touch the target at all -> no target leakage possible.

  3. TARGET (mean) encoding for high-cardinality columns where we want the extra
     signal. This DOES use the target, so it is the single most leakage-prone
     transform in the whole pipeline. We defend with two mechanisms:

       (a) OUT-OF-FOLD encoding on the training set. A row's encoded value is
           computed from the OTHER folds, never from its own row. Without this,
           each row effectively sees its own label and the model overfits
           spectacularly (train AUC ~0.99, test AUC ~0.6).

       (b) SMOOTHING toward the global mean. A country seen 3 times should not
           get a confident 100% cancellation rate. The smoothing parameter `m`
           controls how many observations are needed before we trust the level's
           own mean over the prior.

  All encoders are FIT ON TRAIN ONLY and then applied to test. Unseen test levels
  fall back to the global training mean (target enc) or 0 (frequency enc).

Run:  python src/features/encode.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("encode")

TRAIN_IN = "data/processed/train_featured.csv"
TEST_IN = "data/processed/test_featured.csv"
TRAIN_OUT = "data/processed/train_encoded.csv"
TEST_OUT = "data/processed/test_encoded.csv"
ARTIFACT = "artifacts/encoding_maps.json"

TARGET = "is_canceled"

ONEHOT_COLS = [
    "hotel",
    "deposit_type",
    "customer_type",
    "market_segment",
    "lead_time_bucket",
]
HIGH_CARD_COLS = ["country", "agent", "company"]

# Columns we drop after encoding (raw month string is replaced by cyclical features).
DROP_AFTER = ["arrival_date_month"]


def one_hot(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode, aligning test columns to the train schema.

    Alignment matters: if a level appears in test but not train, naive get_dummies
    would create a column the model never saw. We reindex test to the TRAIN
    columns, filling missing with 0 — the same thing that must happen in
    production when an unseen category arrives.
    """
    train_d = pd.get_dummies(train, columns=cols, prefix=cols, dtype=int)
    test_d = pd.get_dummies(test, columns=cols, prefix=cols, dtype=int)

    new_cols = [c for c in train_d.columns if c not in train.columns]
    test_d = test_d.reindex(columns=train_d.columns, fill_value=0)

    log.info("One-hot: %s -> %d new columns", cols, len(new_cols))
    return train_d, test_d


def frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Frequency encoding: level -> its relative frequency in TRAIN.

    Never touches the target, so it cannot leak label information. Unseen test
    levels get 0 (correctly signalling 'never seen in training').
    """
    train = train.copy()
    test = test.copy()
    maps: dict[str, dict] = {}

    for col in cols:
        freq = train[col].astype(str).value_counts(normalize=True)
        maps[col] = {str(k): float(v) for k, v in freq.items()}
        train[f"{col}_freq"] = train[col].astype(str).map(freq).fillna(0.0)
        test[f"{col}_freq"] = test[col].astype(str).map(freq).fillna(0.0)
        log.info("Frequency-encoded %s (%d levels)", col, len(freq))

    return train, test, maps


def _smoothed_means(stats: pd.DataFrame, prior: float, m: float) -> pd.Series:
    """Bayesian-style smoothing toward the global prior.

        encoded = (count * level_mean + m * prior) / (count + m)

    With count >> m the level's own mean dominates; with count << m we fall back
    to the prior. m is 'how many observations before I trust this level'.
    """
    return (stats["sum"] + m * prior) / (stats["count"] + m)


def target_encode_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    n_splits: int = 5,
    m: float = 20.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Out-of-fold smoothed target encoding — the leakage-safe version.

    TRAIN: for each fold, compute level means on the OTHER folds and apply them to
    this fold. No row ever contributes to its own encoded value.
    TEST:  computed once from the FULL training set (test rows never influence it).
    """
    train = train.copy()
    test = test.copy()
    prior = float(train[TARGET].mean())
    maps: dict[str, dict] = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        key = train[col].astype(str)
        oof = pd.Series(np.nan, index=train.index, dtype=float)

        for tr_idx, val_idx in skf.split(train, train[TARGET]):
            fold_stats = (
                train.iloc[tr_idx]
                .assign(_k=key.iloc[tr_idx])
                .groupby("_k")[TARGET]
                .agg(["sum", "count"])
            )
            fold_prior = float(train.iloc[tr_idx][TARGET].mean())
            enc = _smoothed_means(fold_stats, fold_prior, m)
            oof.iloc[val_idx] = key.iloc[val_idx].map(enc).to_numpy()

        train[f"{col}_te"] = oof.fillna(prior)

        # Test encoding: fit on the FULL training set once.
        full_stats = train.assign(_k=key).groupby("_k")[TARGET].agg(["sum", "count"])
        full_enc = _smoothed_means(full_stats, prior, m)
        test[f"{col}_te"] = test[col].astype(str).map(full_enc).fillna(prior)

        maps[col] = {"prior": prior, "m": m, "n_levels": int(full_enc.shape[0])}
        log.info(
            "Target-encoded %s out-of-fold (%d levels, m=%.0f, prior=%.3f)",
            col,
            full_enc.shape[0],
            m,
            prior,
        )

    return train, test, maps


def encode_all(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full encoding pipeline, train-fit then test-applied throughout."""
    artifacts: dict = {}

    train, test, freq_maps = frequency_encode(train, test, HIGH_CARD_COLS)
    artifacts["frequency"] = freq_maps

    train, test, te_maps = target_encode_oof(train, test, HIGH_CARD_COLS)
    artifacts["target_encoding"] = te_maps

    # Raw high-cardinality columns are now represented numerically; drop the
    # originals so no string columns reach the model.
    train = train.drop(columns=HIGH_CARD_COLS)
    test = test.drop(columns=HIGH_CARD_COLS)

    train, test = one_hot(train, test, ONEHOT_COLS)

    drop_now = [c for c in DROP_AFTER if c in train.columns]
    train = train.drop(columns=drop_now)
    test = test.drop(columns=drop_now)

    return train, test, artifacts


if __name__ == "__main__":
    import os

    train = pd.read_csv(TRAIN_IN)
    test = pd.read_csv(TEST_IN)
    log.info("Loaded featured train=%d, test=%d", len(train), len(test))

    train_e, test_e, artifacts = encode_all(train, test)

    # Sanity: no non-numeric columns should survive.
    non_numeric = train_e.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if non_numeric:
        log.warning("Non-numeric columns remain: %s", non_numeric)
    else:
        log.info("All columns numeric — model-ready ✔")

    assert list(train_e.columns) == list(test_e.columns), "train/test schema mismatch"
    log.info("Schema aligned: %d columns in both splits", train_e.shape[1])

    os.makedirs("artifacts", exist_ok=True)
    with open(ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    train_e.to_csv(TRAIN_OUT, index=False)
    test_e.to_csv(TEST_OUT, index=False)
    log.info("Wrote %s, %s and %s", TRAIN_OUT, TEST_OUT, ARTIFACT)
