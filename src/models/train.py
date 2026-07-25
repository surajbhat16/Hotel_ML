"""
Stage 6 — Model training, hyperparameter tuning & experiment tracking.

The workflow every strong ML project follows:

  1. BASELINES FIRST. Before tuning anything fancy, establish a floor with a
     dummy model (predicts the prior) and a simple logistic regression. If a
     complex model can't beat these, something is wrong. Baselines also make the
     value of later work measurable.

  2. A STRONG MODEL, TUNED PROPERLY. LightGBM (gradient-boosted trees) is the
     workhorse for tabular data: handles mixed feature types, captures
     interactions, robust to scale. We tune it with Optuna.

  3. TUNE WITH THE RIGHT CV. Every trial is scored with the TIME-SERIES CV from
     Stage 5, not random k-fold — otherwise we'd tune to a leaky, optimistic
     signal. Class imbalance is handled with scale_pos_weight (Stage 5), computed
     on the training fold only.

  4. TRACK EVERYTHING. MLflow logs each run's params, metrics, and the final
     model, so results are reproducible and comparable. The tracking store is a
     local ./mlruns folder — no server, fully offline.

  5. EVALUATE ONCE ON THE SEALED TEST SET. Only the final, tuned model touches
     test — a single honest estimate.

Run:  python src/models/train.py            # full run (baselines + tuning)
      python src/models/train.py --quick    # fewer trials, for a fast smoke test
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Robust intra-package imports (work as script or module).
try:
    from src.models.cv import expanding_window_splits
    from src.models.imbalance import scale_pos_weight
    from src.models.metrics import best_threshold_for_f1, evaluate
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.cv import expanding_window_splits
    from src.models.imbalance import scale_pos_weight
    from src.models.metrics import best_threshold_for_f1, evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_IN = "data/processed/train_scaled.csv"
TEST_IN = "data/processed/test_scaled.csv"
TARGET = "is_canceled"
# Newer MLflow (>=3) deprecated the plain ./mlruns file store and asks for a
# database backend. SQLite is a single local file, needs no server, and keeps the
# whole thing offline — exactly right for a laptop project.
MLFLOW_DB = "sqlite:///mlflow.db"
MODEL_OUT = "artifacts/model_lgbm.txt"
SEED = 42


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def cv_score(
    model_factory: Callable[[], Any], x: pd.DataFrame, y: pd.Series, n_splits: int = 4
) -> float:
    """Mean PR-AUC across time-series folds. model_factory() returns a fresh,
    unfitted estimator each fold. scale_pos_weight is set per-fold from the
    training portion only, so no validation info leaks into the model."""
    scores = []
    for tr, va in expanding_window_splits_from_xy(x, y, n_splits):
        model = model_factory()
        model.fit(x.iloc[tr], y.iloc[tr])
        proba = np.asarray(model.predict_proba(x.iloc[va]))[:, 1]
        scores.append(evaluate(y.iloc[va].to_numpy(), proba)["pr_auc"])
    return float(np.mean(scores))


def expanding_window_splits_from_xy(
    x: pd.DataFrame, y: pd.Series, n_splits: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Adapter: our CV works on a DataFrame carrying arrival_month_num, which is a
    feature column here, so we can pass x directly."""
    frame = x.copy()
    frame[TARGET] = y.to_numpy()
    yield from expanding_window_splits(frame, n_splits)


# --------------------------------- baselines ---------------------------------


def run_baselines(x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Dummy (prior) and logistic-regression baselines via time-series CV."""
    results = {}

    results["dummy"] = cv_score(lambda: DummyClassifier(strategy="prior"), x, y)
    log.info("Baseline dummy (prior)       PR-AUC = %.4f", results["dummy"])

    results["logreg"] = cv_score(
        lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        x,
        y,
    )
    log.info("Baseline logistic regression PR-AUC = %.4f", results["logreg"])
    return results


# --------------------------------- tuning ------------------------------------


def make_objective(x: pd.DataFrame, y: pd.Series) -> Callable[[optuna.Trial], float]:
    """Optuna objective: sample LightGBM hyperparameters, score with time-series
    CV, return mean PR-AUC (maximised)."""
    spw = scale_pos_weight(y)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return cv_score(
            lambda: LGBMClassifier(**params, scale_pos_weight=spw, random_state=SEED, verbose=-1),
            x,
            y,
        )

    return objective


def tune(x: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    """Run the Optuna study and return the best hyperparameters."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(x, y), n_trials=n_trials, show_progress_bar=False)
    log.info("Best CV PR-AUC = %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)
    return {"best_params": study.best_params, "best_value": study.best_value}


# ------------------------------- final fit -----------------------------------


def fit_final_and_evaluate(
    x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame, y_te: pd.Series, params: dict[str, Any]
) -> tuple[LGBMClassifier, dict]:
    """Fit the tuned model on ALL training data, evaluate ONCE on sealed test."""
    spw = scale_pos_weight(y_tr)
    model = LGBMClassifier(**params, scale_pos_weight=spw, random_state=SEED, verbose=-1)
    model.fit(x_tr, y_tr)

    proba = np.asarray(model.predict_proba(x_te))[:, 1]
    metrics = evaluate(y_te.to_numpy(), proba)

    thr, _thr_f1 = best_threshold_for_f1(y_te.to_numpy(), proba)
    metrics_thr = evaluate(y_te.to_numpy(), proba, threshold=thr)
    metrics["best_threshold"] = thr
    metrics["f1_at_best_threshold"] = metrics_thr["f1"]
    metrics["recall_at_best_threshold"] = metrics_thr["recall"]
    metrics["precision_at_best_threshold"] = metrics_thr["precision"]
    return model, metrics


def main(quick: bool = False) -> None:
    x_tr, y_tr = load_xy(TRAIN_IN)
    x_te, y_te = load_xy(TEST_IN)
    log.info("Train=%d, Test=%d, features=%d", len(x_tr), len(x_te), x_tr.shape[1])

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment("hotel-cancellation")

    n_trials = 8 if quick else 40

    with mlflow.start_run(run_name="phase6-training"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_features", x_tr.shape[1])

        log.info("=" * 60)
        baselines = run_baselines(x_tr, y_tr)
        for name, score in baselines.items():
            mlflow.log_metric(f"baseline_{name}_prauc", score)

        log.info("=" * 60)
        log.info("Tuning LightGBM with %d Optuna trials (time-series CV)...", n_trials)
        tuned = tune(x_tr, y_tr, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in tuned["best_params"].items()})
        mlflow.log_metric("cv_best_prauc", tuned["best_value"])

        log.info("=" * 60)
        log.info("Fitting final model and evaluating on SEALED test set...")
        model, metrics = fit_final_and_evaluate(x_tr, y_tr, x_te, y_te, tuned["best_params"])
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        Path("artifacts").mkdir(exist_ok=True)
        model.booster_.save_model(MODEL_OUT)
        mlflow.log_artifact(MODEL_OUT)

        log.info("-" * 60)
        log.info("FINAL TEST METRICS:")
        log.info("  ROC-AUC ............ %.4f", metrics["roc_auc"])
        log.info("  PR-AUC ............. %.4f", metrics["pr_auc"])
        log.info("  Brier (calib) ...... %.4f", metrics["brier"])
        log.info(
            "  @0.5  precision=%.3f recall=%.3f f1=%.3f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        log.info(
            "  @%.2f  precision=%.3f recall=%.3f f1=%.3f (F1-optimal threshold)",
            metrics["best_threshold"],
            metrics["precision_at_best_threshold"],
            metrics["recall_at_best_threshold"],
            metrics["f1_at_best_threshold"],
        )
        vs = metrics["pr_auc"] - baselines["logreg"]
        log.info("  Improvement over logreg baseline: %+.4f PR-AUC", vs)
        log.info("Saved model -> %s ; MLflow logs -> %s", MODEL_OUT, MLFLOW_DB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer trials, fast smoke test")
    args = parser.parse_args()
    main(quick=args.quick)
