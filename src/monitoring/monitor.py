"""
Stage 10b — Performance monitoring & retraining policy.

Drift detection watches the INPUTS. This watches the OUTPUTS: is the model still
accurate on recent data where outcomes are now known? A model can pass every
input-drift check and still decay because the RELATIONSHIP between features and
target changed (concept drift), which only shows up in performance.

Two kinds of decay:
  - DATA drift  : P(X) changed          -> caught by drift.py
  - CONCEPT drift: P(y|X) changed        -> caught here, via metric decay

The retraining POLICY combines both signals plus guards, because retraining
blindly on every wobble is its own failure mode (cost, churn, risk of training on
a bad batch). We codify a defensible rule and log why it fired or didn't.
"""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("monitor")

# Thresholds a stakeholder can read and agree to.
# NOTE: the absolute floor is set relative to this project's baseline. On the real
# hotel dataset the model reaches ~0.88 AUC and a floor of 0.55 is appropriate; on
# the synthetic teaching data the baseline is ~0.49, so we key decisions off the
# RELATIVE drop from baseline rather than an absolute floor that the baseline itself
# would violate. Both levers are shown; tune to your data.
PRAUC_ABSOLUTE_FLOOR = 0.35
PRAUC_RELATIVE_DROP = 0.10  # 10% below the training baseline triggers concern
MIN_LABELED_FOR_DECISION = 200  # don't judge performance on a handful of samples


def rolling_metric(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "pr_auc") -> float:
    """Compute a single performance metric on a recent labeled window."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")  # can't score AUC on one class
    if metric == "pr_auc":
        return float(average_precision_score(y_true, y_proba))
    return float(roc_auc_score(y_true, y_proba))


def retraining_decision(
    current_prauc: float,
    baseline_prauc: float,
    drift_summary: dict,
    n_labeled: int,
) -> dict:
    """The retraining policy. Returns a decision + human-readable reasons.

    Fires retraining if EITHER:
      (a) performance decayed below the absolute floor or by more than the
          allowed relative drop from baseline, on ENOUGH labeled data, OR
      (b) significant INPUT drift is present (early warning, before outcomes
          even arrive).
    Otherwise holds. Every branch logs its reasoning."""
    reasons = []
    retrain = False

    perf_reliable = n_labeled >= MIN_LABELED_FOR_DECISION
    if not perf_reliable:
        reasons.append(
            f"performance signal unreliable (only {n_labeled} labeled; "
            f"need {MIN_LABELED_FOR_DECISION})"
        )

    if perf_reliable and not np.isnan(current_prauc):
        rel_drop = (baseline_prauc - current_prauc) / max(baseline_prauc, 1e-9)
        if current_prauc < PRAUC_ABSOLUTE_FLOOR:
            retrain = True
            reasons.append(
                f"PR-AUC {current_prauc:.3f} below absolute floor {PRAUC_ABSOLUTE_FLOOR}"
            )
        elif rel_drop > PRAUC_RELATIVE_DROP:
            retrain = True
            reasons.append(
                f"PR-AUC dropped {rel_drop:.1%} from baseline "
                f"{baseline_prauc:.3f} (limit {PRAUC_RELATIVE_DROP:.0%})"
            )
        else:
            reasons.append(
                f"performance healthy (PR-AUC {current_prauc:.3f} vs baseline {baseline_prauc:.3f})"
            )

    if drift_summary.get("overall_drift") == "significant_drift":
        retrain = True
        reasons.append(f"significant input drift on {drift_summary.get('top_drifted')}")
    elif drift_summary.get("overall_drift") == "moderate_drift":
        reasons.append("moderate input drift (watch, not yet retraining)")

    decision = {
        "retrain": retrain,
        "reasons": reasons,
        "current_prauc": None if np.isnan(current_prauc) else round(current_prauc, 4),
        "baseline_prauc": round(baseline_prauc, 4),
        "drift": drift_summary.get("overall_drift"),
    }
    log.info("Retraining decision: %s", "RETRAIN" if retrain else "HOLD")
    for r in reasons:
        log.info("  - %s", r)
    return decision
