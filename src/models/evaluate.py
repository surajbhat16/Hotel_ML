"""
Stage 7a — Deep evaluation & probability calibration.

Phase 6 gave headline metrics. This phase asks the harder questions a stakeholder
actually cares about:

  1. WHERE does the model separate the classes? (ROC & precision-recall curves)
  2. Are the PROBABILITIES trustworthy, or just the ranking? (calibration curve +
     Brier score, then a calibration fix if needed.)
  3. What THRESHOLD should we actually deploy at, given the business cost of a
     missed cancellation vs a false alarm? (cost curve.)

Why calibration matters here specifically: the downstream dynamic-pricing model
(Phase 12) multiplies P(cancel) by revenue. If the model says 0.8 but the true
rate is 0.5, every price built on it is wrong. A model can have great AUC (perfect
ranking) yet terrible calibration, so we check and fix it explicitly.

Everything is computed on the SEALED test set, using the model trained in Phase 6.
This module only READS the saved model; it never retrains on test.

Run:  python src/models/evaluate.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("evaluate")

TRAIN_IN = "data/processed/train_scaled.csv"
TEST_IN = "data/processed/test_scaled.csv"
MODEL_IN = "artifacts/model_lgbm.txt"
REPORT_OUT = "artifacts/evaluation_report.json"
TARGET = "is_canceled"


def load_model_and_data() -> tuple:
    """Load the saved LightGBM booster and the sealed test set."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=MODEL_IN)
    test = pd.read_csv(TEST_IN)
    x_te = test.drop(columns=[TARGET])
    y_te = test[TARGET].to_numpy()
    proba = booster.predict(x_te)
    return booster, x_te, y_te, np.asarray(proba)


def roc_points(y: np.ndarray, p: np.ndarray) -> dict:
    """ROC curve coordinates + AUC, downsampled for compact storage."""
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int)
    # numpy renamed trapz -> trapezoid in 2.0; support both.
    trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    auc = float(trapz(tpr, fpr))
    log.info("ROC AUC (trapezoid) = %.4f", auc)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": auc}


def pr_points(y: np.ndarray, p: np.ndarray) -> dict:
    """Precision-recall curve — the more informative view under imbalance."""
    precision, recall, _ = precision_recall_curve(y, p)
    idx = np.linspace(0, len(precision) - 1, min(len(precision), 100)).astype(int)
    log.info("PR curve: %d points, baseline precision = %.3f", len(precision), y.mean())
    return {
        "precision": precision[idx].tolist(),
        "recall": recall[idx].tolist(),
        "baseline": float(y.mean()),
    }


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability curve + Brier. If predicted 0.7 really means 70% cancel, the
    curve hugs the diagonal."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(y, p))
    # Expected Calibration Error: average gap between confidence and accuracy.
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    log.info("Brier = %.4f, ECE = %.4f (lower is better)", brier, ece)
    return {
        "mean_predicted": mean_pred.tolist(),
        "fraction_positive": frac_pos.tolist(),
        "brier": brier,
        "ece": ece,
    }


def calibrate_isotonic(y_val: np.ndarray, p_val: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Fit isotonic regression on a validation split, apply to test probabilities.

    Isotonic is a flexible, monotonic remap of predicted->true probability. It's
    fit on data the model didn't train on (here a slice of train used as a
    calibration set) so it doesn't leak test labels.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return np.asarray(iso.predict(p_test))


def cost_curve(y: np.ndarray, p: np.ndarray, fn_cost: float = 5.0, fp_cost: float = 1.0) -> dict:
    """Total business cost as a function of threshold.

    A missed cancellation (false negative) usually costs more than a false alarm
    (false positive) — an empty room you didn't rebook vs an unnecessary follow-up.
    We sweep thresholds and find the cost-minimising operating point.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    costs = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        costs.append(fn * fn_cost + fp * fp_cost)
    best_i = int(np.argmin(costs))
    log.info(
        "Cost-optimal threshold = %.2f (FN cost=%.1f, FP cost=%.1f)",
        thresholds[best_i],
        fn_cost,
        fp_cost,
    )
    return {
        "thresholds": thresholds.tolist(),
        "costs": costs,
        "best_threshold": float(thresholds[best_i]),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
    }


def main() -> None:
    booster, _x_te, y_te, proba = load_model_and_data()
    log.info("Loaded model + sealed test (%d rows)", len(y_te))
    log.info("=" * 60)

    report: dict = {}
    report["roc"] = roc_points(y_te, proba)
    report["pr"] = pr_points(y_te, proba)
    log.info("=" * 60)
    report["calibration_before"] = calibration_report(y_te, proba)

    # Calibrate using a slice of TRAIN as the calibration set (never test labels).
    train = pd.read_csv(TRAIN_IN)
    booster2 = booster
    cal = train.sample(frac=0.2, random_state=0)
    p_cal = np.asarray(booster2.predict(cal.drop(columns=[TARGET])))
    proba_cal = calibrate_isotonic(cal[TARGET].to_numpy(), p_cal, proba)
    log.info("After isotonic calibration:")
    report["calibration_after"] = calibration_report(y_te, proba_cal)

    log.info("=" * 60)
    report["cost"] = cost_curve(y_te, proba)

    Path("artifacts").mkdir(exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", REPORT_OUT)

    b_before = report["calibration_before"]["brier"]
    b_after = report["calibration_after"]["brier"]
    log.info(
        "Calibration Brier: %.4f -> %.4f (%s)",
        b_before,
        b_after,
        "improved" if b_after < b_before else "no gain",
    )


if __name__ == "__main__":
    main()
