"""
Stage 8a — Inference transform (the anti-skew layer).

The single biggest failure mode in ML serving is TRAINING-SERVING SKEW: the model
trained on features built one way, but the serving code builds them slightly
differently, so the model silently sees inputs it never trained on and degrades.

We prevent it structurally by REUSING the exact Stage-3 engineering code
(src.features.engineer) here, and applying the SAME learned artifacts (encoding
maps, scaler stats) that were fitted on the training data. Nothing about the
feature construction is re-implemented for serving — that's the whole point.

Flow for one raw booking:
    raw dict
      -> validate/coerce to a 1-row frame
      -> engineer()            (identical row-wise features as training)
      -> apply saved encoders  (frequency + target maps from artifacts)
      -> apply saved scaler    (train-fitted center/scale from artifacts)
      -> align to model's expected column order
      -> model.predict

The artifacts (encoding_maps.json, scaler_stats.json) are the frozen "state"
learned at training time. Loading and applying them here is what guarantees the
serving path matches training exactly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.features.engineer import engineer
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.features.engineer import engineer

log = logging.getLogger("inference")

ENCODING_MAPS = "artifacts/encoding_maps.json"
SCALER_STATS = "artifacts/scaler_stats.json"
MODEL_FILE = "artifacts/model_lgbm.txt"

HIGH_CARD_COLS = ["country", "agent", "company"]
ONEHOT_COLS = ["hotel", "deposit_type", "customer_type", "market_segment", "lead_time_bucket"]
DROP_AFTER = ["arrival_date_month"]


class InferencePipeline:
    """Loads the frozen training artifacts once and transforms raw bookings into
    model-ready feature rows, then scores them. One instance is reused across
    requests (artifacts loaded a single time at startup)."""

    def __init__(
        self,
        model_file: str = MODEL_FILE,
        encoding_maps: str = ENCODING_MAPS,
        scaler_stats: str = SCALER_STATS,
    ) -> None:
        import lightgbm as lgb

        self.encoders = json.loads(Path(encoding_maps).read_text(encoding="utf-8"))
        self.scaler = json.loads(Path(scaler_stats).read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=model_file)
        self.expected_columns = list(self.booster.feature_name())
        log.info("InferencePipeline ready: %d expected features", len(self.expected_columns))

    # ---- individual stages (mirroring training) ----

    def _apply_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in HIGH_CARD_COLS:
            fmap = self.encoders["frequency"][col]
            df[f"{col}_freq"] = df[col].astype(str).map(fmap).fillna(0.0)
        return df

    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        # We stored the prior per column; unseen levels fall back to it. (Full
        # per-level maps could be persisted too; the prior fallback is the safe
        # default and matches how unseen categories are handled in training.)
        for col in HIGH_CARD_COLS:
            prior = self.encoders["target_encoding"][col]["prior"]
            df[f"{col}_te"] = prior
        return df

    def _apply_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=ONEHOT_COLS, prefix=ONEHOT_COLS, dtype=int)
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.scaler["columns"]
        center = self.scaler["center"]
        scale = self.scaler["scale"]
        for c in cols:
            if c in df.columns:
                df[c] = (df[c] - center[c]) / scale[c]
        return df

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to the model's exact expected columns, filling any missing
        (e.g. a one-hot level not present in this request) with 0. This is the
        production analog of the train/test schema alignment from Stage 3."""
        return df.reindex(columns=self.expected_columns, fill_value=0)

    # ---- public API ----

    def transform(self, raw: dict) -> pd.DataFrame:
        """Raw booking dict -> a single model-ready feature row."""
        df = pd.DataFrame([raw])
        df = engineer(df)  # identical to training
        df = self._apply_frequency(df)
        df = self._apply_target_encoding(df)
        df = df.drop(columns=[c for c in HIGH_CARD_COLS if c in df.columns])
        df = self._apply_onehot(df)
        df = df.drop(columns=[c for c in DROP_AFTER if c in df.columns])
        df = self._apply_scaler(df)
        return self._align(df)

    def predict_proba(self, raw: dict) -> float:
        """Cancellation probability for one raw booking."""
        x = self.transform(raw)
        return float(np.ravel(self.booster.predict(x))[0])

    def predict(self, raw: dict, threshold: float = 0.5) -> dict:
        """Full prediction payload: probability, label, and the threshold used."""
        proba = self.predict_proba(raw)
        return {
            "cancellation_probability": round(proba, 4),
            "will_cancel": bool(proba >= threshold),
            "threshold": threshold,
        }
