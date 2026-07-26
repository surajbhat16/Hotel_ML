"""
Stage 7b — Model explainability with SHAP.

A cancellation model that a business can't interrogate won't be trusted or
deployed. SHAP (SHapley Additive exPlanations) answers two questions:

  GLOBAL:  which features drive the model overall, and in which direction?
  LOCAL:   for THIS specific booking, why did the model predict cancel / not?

SHAP values come from cooperative game theory: each feature's contribution to a
prediction is its average marginal contribution across all possible orderings of
features. The key property is additivity — for any single prediction, the SHAP
values of all features plus the base rate sum exactly to the model's output. That
makes the explanation faithful, not a post-hoc guess.

For tree models we use TreeExplainer, which computes exact SHAP values efficiently.

This module READS the saved model and explains it on test data. It writes a JSON
summary (global importances + a few worked local examples) so the results are
reproducible without a notebook.

Run:  python src/models/explain.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("explain")

MODEL_IN = "artifacts/model_lgbm.txt"
TEST_IN = "data/processed/test_scaled.csv"
REPORT_OUT = "artifacts/shap_report.json"
TARGET = "is_canceled"


def load() -> tuple:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=MODEL_IN)
    test = pd.read_csv(TEST_IN)
    x = test.drop(columns=[TARGET])
    y = test[TARGET].to_numpy()
    return booster, x, y


def compute_shap(
    booster: object, x: pd.DataFrame, sample: int = 2000, seed: int = 0
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values on a sample (for speed). Returns (shap_values, x_sample)."""
    import shap

    x_s = x.sample(min(sample, len(x)), random_state=seed)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_s)
    # LightGBM binary can return a list [class0, class1]; take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    log.info("Computed SHAP for %d rows x %d features", *x_s.shape)
    return np.asarray(shap_values), x_s


def global_importance(shap_values: np.ndarray, x_s: pd.DataFrame, top: int = 15) -> pd.Series:
    """Mean |SHAP| per feature = overall importance, direction-agnostic."""
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=x_s.columns)
    imp = imp.sort_values(ascending=False)
    log.info("Top %d features by mean |SHAP|:\n%s", top, imp.head(top).round(4).to_string())
    return imp


def signed_effect(shap_values: np.ndarray, x_s: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """For the top features, does a HIGH value push toward cancel or stay?

    We correlate each feature's value with its SHAP value: positive => higher
    feature value increases predicted cancellation probability.
    """
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top]
    rows = []
    for j in order:
        col = x_s.columns[j]
        fv = x_s[col].to_numpy()
        sv = shap_values[:, j]
        # Guard against constant columns.
        direction = float(np.corrcoef(fv, sv)[0, 1]) if fv.std() > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "mean_abs_shap": float(np.abs(sv).mean()),
                "direction": "↑cancel" if direction > 0 else "↓cancel",
                "value_shap_corr": round(direction, 3),
            }
        )
    df = pd.DataFrame(rows)
    log.info("Signed effects (top features):\n%s", df.to_string(index=False))
    return df


def local_explanation(
    booster: object, shap_values: np.ndarray, x_s: pd.DataFrame, row: int
) -> dict:
    """Explain a single booking: the features pushing it toward / away from cancel."""
    import shap

    explainer = shap.TreeExplainer(booster)
    base = float(np.ravel(explainer.expected_value)[-1])
    contribs = pd.Series(shap_values[row], index=x_s.columns).sort_values(
        key=np.abs, ascending=False
    )
    top = contribs.head(6)
    pred_logit = base + shap_values[row].sum()
    return {
        "row_index": int(x_s.index[row]),
        "base_value": round(base, 4),
        "prediction_logit": round(float(pred_logit), 4),
        "top_contributions": {k: round(float(v), 4) for k, v in top.items()},
    }


def main() -> None:
    booster, x, _y = load()
    shap_values, x_s = compute_shap(booster, x)
    log.info("=" * 60)

    imp = global_importance(shap_values, x_s)
    log.info("=" * 60)
    signed = signed_effect(shap_values, x_s)
    log.info("=" * 60)

    # A few worked local examples: highest-risk and lowest-risk in the sample.
    row_scores = shap_values.sum(axis=1)
    high = int(np.argmax(row_scores))
    low = int(np.argmin(row_scores))
    log.info("Local example — HIGH risk booking:")
    high_ex = local_explanation(booster, shap_values, x_s, high)
    log.info("  %s", high_ex["top_contributions"])
    log.info("Local example — LOW risk booking:")
    low_ex = local_explanation(booster, shap_values, x_s, low)
    log.info("  %s", low_ex["top_contributions"])

    report = {
        "global_importance": imp.head(20).round(5).to_dict(),
        "signed_effects": signed.to_dict(orient="records"),
        "local_high_risk": high_ex,
        "local_low_risk": low_ex,
    }
    Path("artifacts").mkdir(exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", REPORT_OUT)


if __name__ == "__main__":
    main()
