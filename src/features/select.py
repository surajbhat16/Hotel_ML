"""
Stage 4b — Feature selection.

Why select at all when we only have ~58 features? Three reasons that scale to any
size: (1) redundant/irrelevant features add variance and can hurt generalisation,
(2) fewer features means faster training, cheaper serving, and easier monitoring,
(3) a smaller, understood feature set is far easier to explain and debug.

The three families — every interview expects you to know all three:

  FILTER    — rank features by a statistic computed independently of any model.
              Fast, model-agnostic, but blind to feature interactions.
              Here: mutual information and ANOVA F-test.

  EMBEDDED  — selection happens *inside* model training. L1 (Lasso) drives
              coefficients to zero; tree models expose impurity/gain importance.
              Captures interactions, but is tied to that model's inductive bias.
              Here: L1-logistic and LightGBM importance.

  WRAPPER   — repeatedly train a model on feature subsets and keep what helps.
              Most powerful (sees interactions + the actual metric) and most
              expensive. Here: Recursive Feature Elimination (RFE).

ALL selection is fit on TRAIN ONLY. Selecting features using the test set is
leakage — the choice of features would be informed by the data you're meant to be
evaluating against.

Run:  python src/features/select.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("select")

TRAIN_IN = "data/processed/train_scaled.csv"
RANK_ARTIFACT = "artifacts/feature_rankings.json"
TARGET = "is_canceled"


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def filter_mutual_info(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual information: how much knowing a feature reduces uncertainty about
    the target. Captures non-linear dependence; zero iff independent."""
    mi = mutual_info_classif(x, y, random_state=42)
    s = pd.Series(mi, index=x.columns).sort_values(ascending=False)
    log.info("Filter (mutual info) top 8:\n%s", s.head(8).round(4).to_string())
    return s


def filter_anova_f(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ANOVA F-test: ratio of between-class to within-class variance. Linear, but
    a fast, classic screen."""
    selector = SelectKBest(score_func=f_classif, k="all").fit(x, y)
    s = pd.Series(selector.scores_, index=x.columns).sort_values(ascending=False)
    log.info("Filter (ANOVA F) top 8:\n%s", s.head(8).round(2).to_string())
    return s


def embedded_l1(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """L1-regularised logistic regression: the penalty forces weak coefficients to
    exactly zero, so non-zero magnitude is a selection signal.

    We build the estimator in a version-safe way: newer scikit-learn (>=1.8)
    deprecates ``penalty='l1'`` in favour of ``l1_ratio``, while older versions
    only accept ``penalty``. We try the modern signature first and fall back to
    the classic one, so the module runs warning-free across versions.
    """
    try:
        # Modern API (sklearn >= 1.8): elastic-net param, l1_ratio=1 == pure L1.
        model = LogisticRegression(
            solver="saga", l1_ratio=1.0, C=0.1, max_iter=2000, random_state=42
        )
        model.fit(x, y)
    except TypeError:
        # Classic API (older sklearn).
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1, max_iter=1000, random_state=42
        )
        model.fit(x, y)
    s = pd.Series(np.abs(model.coef_[0]), index=x.columns).sort_values(ascending=False)
    n_zero = int((s == 0).sum())
    log.info(
        "Embedded (L1) drove %d/%d coefficients to zero. Top 8:\n%s",
        n_zero,
        len(s),
        s.head(8).round(4).to_string(),
    )
    return s


def embedded_tree_importance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Tree gain importance from a fast gradient-boosted model. Captures
    interactions natively. Import lazily so the module loads without lightgbm."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.warning("lightgbm not installed — skipping tree importance.")
        return pd.Series(dtype=float)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ).fit(x, y)
    s = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    log.info("Embedded (tree gain) top 8:\n%s", s.head(8).round(1).to_string())
    return s


def wrapper_rfe(x: pd.DataFrame, y: pd.Series, n_features: int = 20) -> pd.Series:
    """Recursive Feature Elimination: fit, drop the weakest, repeat. Returns a
    rank (1 = kept/most important). Uses a cheap linear estimator as the core."""
    estimator = LogisticRegression(max_iter=2000, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features, step=0.1).fit(x, y)
    rank = pd.Series(rfe.ranking_, index=x.columns).sort_values()
    kept = rank[rank == 1].index.tolist()
    log.info("Wrapper (RFE) kept %d features: %s", len(kept), kept)
    return rank


def consensus_ranking(rankings: dict[str, pd.Series], top_k: int = 25) -> pd.DataFrame:
    """Combine methods into a robust consensus. Each method votes by ranking its
    features; we average the (normalised) ranks. A feature strong across diverse
    methods is a safer keep than one that only one method loves."""
    frames = []
    for name, s in rankings.items():
        if s.empty:
            continue
        # Higher score = better -> rank descending; RFE is already a rank (lower better).
        r = s.rank(ascending=True) if name == "rfe" else s.rank(ascending=False)
        frames.append(r.rename(name))

    combined = pd.concat(frames, axis=1)
    combined["mean_rank"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_rank")
    log.info("Consensus top %d features:\n%s", top_k, combined.head(top_k).index.tolist())
    return combined


if __name__ == "__main__":
    import os

    train = pd.read_csv(TRAIN_IN)
    x, y = _xy(train)
    log.info("Loaded scaled train: %d rows, %d features", len(x), x.shape[1])

    rankings = {
        "mutual_info": filter_mutual_info(x, y),
        "anova_f": filter_anova_f(x, y),
        "l1": embedded_l1(x, y),
        "tree": embedded_tree_importance(x, y),
        "rfe": wrapper_rfe(x, y, n_features=20),
    }

    consensus = consensus_ranking(rankings, top_k=25)

    os.makedirs("artifacts", exist_ok=True)
    out = {
        "consensus_order": consensus.index.tolist(),
        "top_25": consensus.head(25).index.tolist(),
        "methods": {k: v.head(15).round(5).to_dict() for k, v in rankings.items() if not v.empty},
    }
    with open(RANK_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", RANK_ARTIFACT)
