"""
Stage 4a — Feature scaling.

Scaling matters for distance- and gradient-based models (logistic regression,
SVMs, k-NN, neural nets) where features on larger numeric scales otherwise
dominate. Tree ensembles (our likely final model) are scale-invariant, so we keep
scaling as a SEPARATE, optional step rather than baking it into the features —
that way we can feed raw features to trees and scaled features to linear baselines
from the same pipeline.

THE ONE RULE: the scaler is FIT ON TRAIN ONLY, then applied to test. Fitting on
the full dataset (or worse, fitting separately on test) leaks the test
distribution into the transform and inflates evaluation. The fitted scaler is
serialised so production applies the exact same train-learned statistics.

Three scalers, matched to distribution shape:
  - StandardScaler  : (x - mean) / std. Default for roughly symmetric features.
  - RobustScaler    : (x - median) / IQR. Resistant to outliers; good for the
                      long-tailed features (lead_time, adr) even after log.
  - MinMaxScaler    : squashes to [0, 1]. Useful when a bounded range is needed.

We do NOT scale binary / one-hot / already-bounded columns — scaling a 0/1
indicator is meaningless and just obscures it. We detect and skip them.

Run:  python src/features/scale.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("scale")

TRAIN_IN = "data/processed/train_encoded.csv"
TEST_IN = "data/processed/test_encoded.csv"
TRAIN_OUT = "data/processed/train_scaled.csv"
TEST_OUT = "data/processed/test_scaled.csv"
STATS_ARTIFACT = "artifacts/scaler_stats.json"

TARGET = "is_canceled"


def detect_binary_columns(df: pd.DataFrame) -> list[str]:
    """A column is 'binary/indicator' if it only holds {0, 1}. These should not
    be scaled — one-hot and boolean flags carry meaning in their 0/1 form."""
    binary = []
    for col in df.columns:
        vals = set(pd.unique(df[col].dropna()))
        if vals <= {0, 1}:
            binary.append(col)
    return binary


def columns_to_scale(df: pd.DataFrame) -> list[str]:
    """Continuous numeric columns only: exclude the target and binaries."""
    binary = set(detect_binary_columns(df))
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET and c not in binary
    ]


def fit_scaler(
    train: pd.DataFrame, cols: list[str], kind: str = "robust"
) -> RobustScaler | StandardScaler:
    """Fit the chosen scaler on TRAIN columns only. Returns the fitted object."""
    scaler = RobustScaler() if kind == "robust" else StandardScaler()
    scaler.fit(train[cols])
    log.info("Fitted %sScaler on %d continuous columns (train only)", kind.capitalize(), len(cols))
    return scaler


def apply_scaler(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaler: RobustScaler | StandardScaler,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform both splits with the TRAIN-fitted scaler."""
    train = train.copy()
    test = test.copy()
    train[cols] = scaler.transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    return train, test


def scale_pipeline(
    train: pd.DataFrame, test: pd.DataFrame, kind: str = "robust"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cols = columns_to_scale(train)
    skipped = [c for c in train.columns if c not in cols and c != TARGET]
    log.info("Scaling %d columns; skipping %d binary/indicator columns", len(cols), len(skipped))

    scaler = fit_scaler(train, cols, kind)
    train_s, test_s = apply_scaler(train, test, scaler, cols)

    # Persist the learned statistics so production can reproduce the transform.
    stats = {
        "kind": kind,
        "columns": cols,
        "center": dict(zip(cols, np.asarray(scaler.center_).tolist(), strict=True))
        if hasattr(scaler, "center_")
        else dict(zip(cols, np.asarray(scaler.mean_).tolist(), strict=True)),
        "scale": dict(zip(cols, np.asarray(scaler.scale_).tolist(), strict=True)),
    }
    return train_s, test_s, stats


if __name__ == "__main__":
    import os

    train = pd.read_csv(TRAIN_IN)
    test = pd.read_csv(TEST_IN)
    log.info("Loaded encoded train=%d, test=%d", len(train), len(test))

    train_s, test_s, stats = scale_pipeline(train, test, kind="robust")

    os.makedirs("artifacts", exist_ok=True)
    with open(STATS_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    train_s.to_csv(TRAIN_OUT, index=False)
    test_s.to_csv(TEST_OUT, index=False)
    log.info("Wrote %s, %s and %s", TRAIN_OUT, TEST_OUT, STATS_ARTIFACT)
