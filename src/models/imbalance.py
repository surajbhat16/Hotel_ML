"""
Stage 5b — Class-imbalance handling.

The target is ~33% positive (2:1). That's moderate, not severe, but enough that a
naive model can lean on the majority class and that raw accuracy is misleading.
Three families of remedy, and — critically — the leakage trap that ruins the most
popular one.

  1. CLASS WEIGHTS — tell the loss function to penalise minority-class mistakes
     more. No data is added or removed. Cheap, leak-proof, and usually the first
     thing to try. Most sklearn/boosting models take class_weight or
     scale_pos_weight.

  2. RESAMPLING —
       * Oversampling (SMOTE): synthesise new minority examples by interpolating
         between neighbours.
       * Undersampling: drop majority examples.
     Powerful but dangerous (see the leakage note below).

  3. THRESHOLD MOVING — train normally, then choose the decision threshold that
     matches the business cost, instead of the default 0.5. Often the single most
     effective and honest lever, because it separates the model from the
     operating point.

======================================================================
THE SMOTE LEAKAGE TRAP (the #1 imbalance interview question)
----------------------------------------------------------------------
SMOTE must be applied to the TRAINING FOLD ONLY, INSIDE cross-validation —
never to the whole dataset before splitting, and never to a validation/test fold.

If you SMOTE before splitting, synthetic minority points are interpolated from
neighbours that may end up on BOTH sides of the split. A validation point can then
be a near-copy of a synthetic training point, so the model is effectively tested
on data it trained on. Reported scores look great and collapse in production.

Correct order, every fold:  split -> fit SMOTE on train fold -> transform train
fold -> validate on the ORIGINAL, untouched validation fold.
======================================================================

Run:  python src/models/imbalance.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("imbalance")

TRAIN_IN = "data/processed/train_scaled.csv"
TARGET = "is_canceled"


def class_weights(y: pd.Series) -> dict[int, float]:
    """Balanced weights: inversely proportional to class frequency, normalised so
    the average weight is ~1. This is what class_weight='balanced' computes."""
    counts = y.value_counts()
    n = len(y)
    k = len(counts)
    weights = {int(c): float(n / (k * cnt)) for c, cnt in counts.items()}
    log.info("Balanced class weights: %s", {c: round(w, 3) for c, w in weights.items()})
    return weights


def scale_pos_weight(y: pd.Series) -> float:
    """The single scalar XGBoost/LightGBM use: negatives / positives."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = neg / pos
    log.info("scale_pos_weight (neg/pos) = %.3f", spw)
    return spw


def smote_on_train_fold_only(
    x_tr: pd.DataFrame, y_tr: pd.Series, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to a SINGLE training fold. This function deliberately takes only
    the training fold — there is no code path that lets it see validation/test
    data, which is the whole point.

    Returns the resampled (x, y). Import is lazy so the module loads without
    imbalanced-learn installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.warning("imbalanced-learn not installed — returning data unchanged.")
        return x_tr, y_tr

    before = y_tr.value_counts().to_dict()
    x_res, y_res = SMOTE(random_state=seed).fit_resample(x_tr, y_tr)
    after = pd.Series(y_res).value_counts().to_dict()
    log.info("SMOTE on train fold: %s -> %s", before, after)
    return x_res, pd.Series(y_res, name=TARGET)


def demonstrate_leakage_trap(df: pd.DataFrame, n_splits: int = 5) -> None:
    """Empirically contrast the WRONG and RIGHT way to combine SMOTE with CV,
    measuring the optimistic bias the wrong way introduces."""
    try:
        from imblearn.over_sampling import SMOTE
        from lightgbm import LGBMClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.warning("Skipping demo — needs imbalanced-learn + lightgbm.")
        return

    # Robust import whether run as `python src/models/imbalance.py` or `-m`.
    try:
        from src.models.cv import expanding_window_splits
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.models.cv import expanding_window_splits

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    # WRONG: SMOTE the whole training set once, THEN cross-validate on resampled data.
    x_all, y_all = SMOTE(random_state=0).fit_resample(x, y)
    wrong_scores = []
    # Re-derive folds on the resampled frame (this is the mistake people make).
    from sklearn.model_selection import KFold

    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=0).split(x_all):
        m = LGBMClassifier(n_estimators=100, random_state=0, verbose=-1)
        m.fit(x_all.iloc[tr], y_all[tr])
        p = np.asarray(m.predict_proba(x_all.iloc[va]))[:, 1]
        wrong_scores.append(roc_auc_score(y_all[va], p))

    # RIGHT: SMOTE inside each fold, on the training portion only; validate on
    # the original untouched fold.
    right_scores = []
    for tr, va in expanding_window_splits(df, n_splits):
        x_tr, y_tr = SMOTE(random_state=0).fit_resample(x.iloc[tr], y.iloc[tr])
        m = LGBMClassifier(n_estimators=100, random_state=0, verbose=-1)
        m.fit(x_tr, y_tr)
        p = np.asarray(m.predict_proba(x.iloc[va]))[:, 1]
        right_scores.append(roc_auc_score(y.iloc[va], p))

    log.info(
        "WRONG (SMOTE before CV) mean AUC = %.3f  <- optimistic / dishonest",
        float(np.mean(wrong_scores)),
    )
    log.info(
        "RIGHT (SMOTE inside fold) mean AUC = %.3f  <- trustworthy", float(np.mean(right_scores))
    )


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_IN)
    y = df[TARGET]
    log.info("Loaded scaled train: %d rows, cancel rate %.3f", len(df), y.mean())
    log.info("=" * 60)
    class_weights(y)
    scale_pos_weight(y)
    log.info("=" * 60)
    log.info("Demonstrating the SMOTE-before-CV leakage trap...")
    demonstrate_leakage_trap(df, n_splits=5)
