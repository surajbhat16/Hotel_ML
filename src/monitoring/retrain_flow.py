"""
Stage 10c — Prefect retraining flow (the orchestrated MLOps loop).

This closes the loop the whole project has been building toward: new data arrives,
we check it for drift, measure performance where outcomes are known, and let a
codified policy decide whether to retrain — all as an orchestrated, schedulable
Prefect flow with automatic retries and observability.

Why Prefect: it doesn't process data itself, it SEQUENCES and SUPERVISES the steps
that do. Each @task is a unit of work with automatic retries; the @flow wires them
into a dependency graph. In production .serve(cron=...) schedules it (e.g. nightly)
and the Prefect UI shows every run, success or failure, with logs.

Run the loop once, locally, no server needed:
    python src/monitoring/retrain_flow.py

Schedule it (optional, needs `prefect server start` in another terminal):
    the .serve(...) call at the bottom registers a cron schedule.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task

try:
    from src.monitoring.drift import detect_drift, summarize
    from src.monitoring.monitor import retraining_decision, rolling_metric
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.monitoring.drift import detect_drift, summarize
    from src.monitoring.monitor import retraining_decision, rolling_metric

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("retrain_flow")

REFERENCE = "data/processed/train.csv"
BASELINE_PRAUC = 0.49  # from the Phase 6 training run (would be read from MLflow)
TARGET = "is_canceled"


@task(retries=2, retry_delay_seconds=5)
def ingest_recent_window(n: int = 4000, drift: bool = False, seed: int = 0) -> pd.DataFrame:
    """Stand-in for 'pull the latest window of live bookings'. With drift=True we
    synthesise a shifted window so the flow can demonstrate catching it. In
    production this reads from the prediction log / feature store."""
    ref = pd.read_csv(REFERENCE)
    window = ref.sample(n, random_state=seed).copy()
    if drift:
        rng = np.random.default_rng(seed)
        window["lead_time"] = window["lead_time"] * 1.6 + rng.normal(20, 5, len(window))
        flip = rng.random(len(window)) < 0.3
        window.loc[flip, "market_segment"] = "Groups"
    log.info("Ingested recent window: %d rows (drift=%s)", len(window), drift)
    return window


@task
def validate_window(window: pd.DataFrame) -> pd.DataFrame:
    """Data-quality gate: reject a window that's obviously broken before it
    poisons any decision (schema, emptiness, absurd values)."""
    assert len(window) > 0, "empty window"
    assert TARGET in window.columns, "missing target column"
    assert window["adr"].between(-1, 10000).all(), "ADR out of sane range"
    log.info("Window validated: %d rows, schema OK", len(window))
    return window


@task(retries=2, retry_delay_seconds=5)
def check_drift(window: pd.DataFrame) -> dict:
    reference = pd.read_csv(REFERENCE)
    report = detect_drift(reference, window)
    summary = summarize(report)
    log.info("Drift: %s (max PSI %.3f)", summary["overall_drift"], summary["max_psi"])
    return summary


@task
def measure_performance(window: pd.DataFrame) -> tuple[float, int]:
    """Score the window with the current model and compute PR-AUC where labels
    exist. In production 'labels' arrive with a delay; here they're present."""
    try:
        from src.serving.inference import InferencePipeline
    except ModuleNotFoundError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.serving.inference import InferencePipeline

    pipe = InferencePipeline()
    raw_cols = [c for c in window.columns if c != TARGET]
    probas = np.array([pipe.predict_proba(r[raw_cols].to_dict()) for _, r in window.iterrows()])
    y = window[TARGET].to_numpy()
    prauc = rolling_metric(y, probas, "pr_auc")
    log.info("Window PR-AUC = %.4f on %d labeled rows", prauc, len(y))
    return prauc, len(y)


@task
def decide(prauc: float, n_labeled: int, drift_summary: dict) -> dict:
    return retraining_decision(prauc, BASELINE_PRAUC, drift_summary, n_labeled)


@task
def trigger_retraining(decision: dict) -> str:
    """In production this would kick off the Phase-6 training pipeline (as a
    sub-flow or a separate job) and register the new model in MLflow. Here we log
    the action so the flow is safe to run repeatedly."""
    if decision["retrain"]:
        log.info(">>> RETRAINING TRIGGERED — would run training pipeline + register model")
        return "retraining_triggered"
    log.info(">>> No retraining needed this cycle")
    return "held"


@flow(name="hotel-drift-monitor-retrain")
def monitoring_flow(inject_drift: bool = False) -> dict:
    """The full monitoring + retraining-decision loop as one orchestrated flow."""
    window = ingest_recent_window(drift=inject_drift)
    window = validate_window(window)
    drift_summary = check_drift(window)
    prauc, n_labeled = measure_performance(window)
    decision = decide(prauc, n_labeled, drift_summary)
    action = trigger_retraining(decision)
    return {"action": action, "decision": decision, "drift": drift_summary}


if __name__ == "__main__":
    log.info("#" * 60)
    log.info("RUN 1: healthy window (no injected drift) -> expect HOLD")
    log.info("#" * 60)
    monitoring_flow(inject_drift=False)

    log.info("#" * 60)
    log.info("RUN 2: drifted window -> expect RETRAIN")
    log.info("#" * 60)
    monitoring_flow(inject_drift=True)

    # To schedule (optional, needs a Prefect server):
    #   monitoring_flow.serve(name="nightly-monitor", cron="0 2 * * *")
