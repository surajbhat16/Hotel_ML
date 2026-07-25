"""
Stage 6 helper — evaluation metrics.

One place that defines how we score every model, so baselines and tuned models are
compared on identical footing. For an imbalanced problem we deliberately lead with
PR-AUC and report accuracy only as a sanity check, never as the headline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return the full metric suite for a set of predicted probabilities.

    - roc_auc / pr_auc are threshold-free ranking metrics (pr_auc = average
      precision, the right primary metric under imbalance).
    - precision/recall/f1 are computed at the given threshold.
    - brier is calibration quality (lower is better) — how well the probabilities
      themselves are trusted, which matters for the pricing model downstream.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return (threshold, f1) maximising F1. This is the
    'threshold moving' lever from Stage 5, applied concretely."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1
