"""
Stage 10a — Data drift detection.

A model is trained on one distribution and deployed against a world that keeps
changing. DRIFT is when the live data stops looking like the training data, and
it's the main reason models silently rot in production. We detect it before it
hurts, by comparing a REFERENCE distribution (the training data) against a
CURRENT window (recent live data).

Two complementary tests, implemented from first principles so the mechanics are
clear (a library like Evidently wraps these, and we show that too):

  PSI (Population Stability Index) — the workhorse for tabular drift.
    Bin the reference, apply the same bins to current, and sum
        (curr% - ref%) * ln(curr% / ref%)
    across bins. It's a symmetric measure of how much probability mass moved.
    Rule of thumb: <0.1 no real shift, 0.1-0.25 moderate, >0.25 significant.

  KS-test (Kolmogorov-Smirnov) — for continuous features. Compares the two
    empirical CDFs and returns the max gap between them plus a p-value. Good for
    detecting shape/location changes a coarse binning might miss.

For CATEGORICAL features, PSI on category frequencies is the natural choice.

DRIFT != BROKEN. Drift is a signal to investigate, not an automatic failure.
Sometimes the world genuinely changed and you retrain; sometimes it's a data-
quality bug upstream. The monitor flags; a human (or a guarded automated policy)
decides.

Run:  python src/monitoring/drift.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("drift")

REFERENCE = "data/processed/train.csv"
TARGET = "is_canceled"

PSI_NO_DRIFT = 0.10
PSI_MODERATE = 0.25


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a continuous feature using quantile bins from the REFERENCE.

    Quantile bins (not equal-width) so each reference bin holds ~equal mass; that
    makes PSI sensitive across the whole range, not just where data is dense.
    A tiny epsilon avoids log(0)/divide-by-zero when a bin empties out.
    """
    eps = 1e-6
    # Bin edges from reference quantiles; unique() guards against duplicate edges.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category frequencies. Categories unseen in reference still count
    via the epsilon floor, so a brand-new category registers as drift."""
    eps = 1e-6
    ref_freq = reference.astype(str).value_counts(normalize=True)
    cur_freq = current.astype(str).value_counts(normalize=True)
    cats = set(ref_freq.index) | set(cur_freq.index)

    psi = 0.0
    for c in cats:
        r = ref_freq.get(c, 0.0) + eps
        k = cur_freq.get(c, 0.0) + eps
        psi += (k - r) * np.log(k / r)
    return float(psi)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test. Returns (statistic, p_value).
    statistic = max distance between the two empirical CDFs."""
    from scipy.stats import ks_2samp

    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def classify_psi(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "no_drift"
    if psi < PSI_MODERATE:
        return "moderate_drift"
    return "significant_drift"


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Run PSI (all features) and KS (numeric features) for each column, returning
    a tidy report sorted by PSI descending."""
    features = features or [c for c in reference.columns if c != TARGET]
    rows = []
    for col in features:
        if col not in current.columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(reference[col])
        if is_numeric:
            psi = psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
            ks_stat, ks_p = ks_test(reference[col].to_numpy(), current[col].to_numpy())
        else:
            psi = psi_categorical(reference[col], current[col])
            ks_stat, ks_p = np.nan, np.nan
        rows.append(
            {
                "feature": col,
                "type": "numeric" if is_numeric else "categorical",
                "psi": round(psi, 4),
                "drift": classify_psi(psi),
                "ks_stat": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
                "ks_pvalue": round(ks_p, 4) if not np.isnan(ks_p) else None,
            }
        )
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def summarize(report: pd.DataFrame) -> dict:
    """Roll the per-feature report into an overall drift verdict."""
    n_sig = int((report["drift"] == "significant_drift").sum())
    n_mod = int((report["drift"] == "moderate_drift").sum())
    max_psi = float(report["psi"].max()) if len(report) else 0.0
    overall = "significant_drift" if n_sig else ("moderate_drift" if n_mod else "no_drift")
    return {
        "overall_drift": overall,
        "n_significant": n_sig,
        "n_moderate": n_mod,
        "max_psi": round(max_psi, 4),
        "top_drifted": report.head(5)["feature"].tolist(),
    }


def _make_drifted_current(reference: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For the demo: synthesise a 'current' window that has genuinely drifted, so
    the detector has something to catch. We push lead_time up and shift the
    market-segment mix (as if booking behaviour changed)."""
    rng = np.random.default_rng(seed)
    cur = reference.sample(4000, random_state=seed).copy()
    cur["lead_time"] = cur["lead_time"] * 1.6 + rng.normal(20, 5, len(cur))
    # Shift categorical mix: bias toward Groups (higher-cancel segment).
    flip = rng.random(len(cur)) < 0.3
    cur.loc[flip, "market_segment"] = "Groups"
    return cur


if __name__ == "__main__":
    reference = pd.read_csv(REFERENCE)
    log.info("Reference (training) rows: %d", len(reference))

    # Case 1: no-drift sanity check — a fresh sample of the SAME distribution.
    same = reference.sample(4000, random_state=1)
    rep_same = detect_drift(reference, same)
    log.info("=" * 60)
    log.info("NO-DRIFT case (same distribution):")
    log.info("  %s", summarize(rep_same))

    # Case 2: injected drift — should be caught.
    drifted = _make_drifted_current(reference)
    rep_drift = detect_drift(reference, drifted)
    log.info("=" * 60)
    log.info("DRIFTED case (lead_time inflated, segment shifted):")
    log.info("  %s", summarize(rep_drift))
    log.info(
        "Top drifted features:\n%s",
        rep_drift[["feature", "psi", "drift"]].head(6).to_string(index=False),
    )
