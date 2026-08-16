# Hotel Booking ML Platform

A hotel booking cancels roughly one time in three. That single fact drives staffing, overbooking policy, and pricing — and it's expensive to get wrong in either direction: hold rooms for bookings that never show, and revenue sits empty; overbook against phantom cancellations, and you're walking a guest at check-in.

This repo is an end-to-end ML platform built around that problem, from raw data to a served, monitored, continuously-evaluated model — plus two downstream systems (demand forecasting and dynamic pricing) that consume its output. It's built solo, runs entirely on open-source tooling, and every design decision below is one I can defend, not one I copied from a tutorial.

**Three problems, one pipeline:**

| Problem | Type | Answers |
|---|---|---|
| Cancellation risk | Binary classification | Will *this* booking cancel? |
| Demand | Time-series regression | How many bookings arrive next month, net of cancellations? |
| Pricing | Constrained optimization | What nightly rate maximizes expected revenue? |

## Results, up front

| Metric | Value |
|---|---|
| Cancellation model — ROC-AUC / PR-AUC (sealed test set) | 0.660 / 0.495 |
| Cancellation model — Brier score, before → after isotonic calibration | 0.229 → 0.206 |
| Cancellation model — calibration error (ECE), before → after | 0.153 → 0.010 |
| Demand forecast — LightGBM MAE vs. seasonal-naive baseline | 437 vs. 1,125 bookings/month (−61%) |
| Serving latency, p95, single booking, full feature pipeline included | ~37 ms |
| Test suite | 92 tests, `ruff` + `mypy --strict` clean |

The cancellation numbers are deliberately unglamorous. A tuned, Optuna-searched LightGBM model essentially *ties* a plain logistic regression baseline on PR-AUC — full analysis in [Known limitations](#known-limitations--whats-next). I kept that result instead of hiding it, because knowing when a fancy model isn't earning its complexity is the actual skill being demonstrated here, not the AUC number.

## How the pieces connect

```
raw synthetic data
      │
      ▼
  clean  ──▶  time-based split  ──▶  EDA + leakage hunt
                                            │
                                            ▼
                          engineer (20 features) ──▶ encode ──▶ scale ──▶ select
                                            │
                                            ▼
                      time-series CV design + imbalance handling (class weights)
                                            │
                                            ▼
              train: dummy + logreg baselines  ──▶  Optuna-tuned LightGBM  ──▶  MLflow
                                            │
                                            ▼
                  evaluate  ──▶  isotonic calibration  ──▶  SHAP explainability
                                            │
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                        ▼
            FastAPI + Docker         streaming consumer      demand forecast
           (on-demand scoring)     (Kafka-shaped simulation)  ──▶ dynamic pricing
                    │                       │
                    └───────────┬───────────┘
                                ▼
                 drift detection + retraining policy
                    (Prefect-orchestrated, schedulable)
```

Every arrow above is a real dependency, not a diagram convenience. The split happens before any statistic is computed, because the imputation and encoding stages fit exclusively on the training side of that split and replay the learned values onto test. The serving layer imports the *same* feature-engineering function the training pipeline uses — not a reimplementation — so a booking scored live goes through byte-identical code to one scored during training. The monitoring flow's performance check loads the same `InferencePipeline` the API serves from. Nothing downstream re-derives a number an upstream stage already computed and could hand off directly; that discipline is the actual point of the project.

## The pipeline, phase by phase

### Data foundation
**Ingestion.** The real Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019) isn't redistributable, so `src/ingestion/generate_dataset.py` regenerates its shape from scratch: 119,690 rows with the same messiness the real data has — 14% missing agent IDs, 95% missing company IDs, a handful of nonsense ADR values including the dataset's infamous €5,400 outlier, 300 injected duplicate rows, and a genuine logistic signal underneath (lead time, deposit type, prior cancellations) so the label isn't noise. It also plants a leakage trap on purpose — a `reservation_status` column that perfectly encodes the outcome — because catching that trap is the first thing the next stage has to do.

**Cleaning.** `src/processing/clean.py` draws a hard line between *structural* fixes (dedup, clipping impossible values, sentinel-filling `agent`/`company` because their absence is meaningful, not missing) and *statistical* fixes (imputing `country`'s mode), because only the first category is safe to apply before any train/test split exists. The leakage column gets dropped here, first, before anything else touches the data.

**Split + EDA.** `src/processing/split.py` splits by arrival date, not randomly — training on the past and testing on the future, because that's the only honest way to evaluate a model that will run in a world where the future hasn't happened yet. `src/processing/eda.py` then runs entirely against the training half, with a systematic leakage scan (flag anything correlating with the target above 0.85) baked in as a repeatable check, not a one-off notebook cell.

### Feature layer
**Engineering.** `src/features/engineer.py` builds 20 features, every one traceable to a specific EDA finding — log transforms for skew, cyclical sin/cos month encoding so December and January register as adjacent instead of maximally distant, an explicit `lead_time × deposit_type` interaction because the EDA showed cancellation risk climbing with lead time at a different rate per deposit type. All of it is row-wise and stateless, which is what makes it safe to run unmodified inside the serving path later.

**Encoding.** `src/features/encode.py` uses three encoding families deliberately matched to cardinality — one-hot for low-cardinality nominals, frequency encoding for `country`/`agent`/`company`, and out-of-fold smoothed target encoding for the same three columns, because naive target encoding is the single easiest way to leak a label into a feature. The out-of-fold mechanism is unit-tested directly: one test asserts a row's encoded value never depends on its own label.

**Scaling + selection.** `src/features/scale.py` fits a `RobustScaler` on train only, skipping every binary/one-hot column since scaling a 0/1 flag is meaningless. `src/features/select.py` runs five independent selection methods — two filter (mutual information, ANOVA F), two embedded (L1-logistic, LightGBM gain), one wrapper (RFE) — and takes their consensus ranking, because trusting any single method's blind spot is how a real feature gets accidentally dropped.

### Modeling
**CV + imbalance.** `src/models/cv.py` implements expanding- and rolling-window time-series cross-validation, because ordinary k-fold on time-ordered data trains on the future to predict the past — a dishonest score that will not survive contact with production. `src/models/imbalance.py` demonstrates the SMOTE-before-CV leakage trap empirically: the same data scored 0.829 AUC when resampled before splitting and 0.628 when resampled correctly inside each fold. The lower number is the one to trust.

**Training.** `src/models/train.py` runs two baselines first (a dummy prior, and class-weighted logistic regression) before touching a gradient-boosted model, because a baseline is what makes a fancy model's contribution measurable instead of assumed. Optuna tunes LightGBM's hyperparameters against the time-series CV, never against a leaky random split. MLflow (SQLite-backed, fully local) logs every run's parameters, metrics, and model artifact.

**Evaluation + explainability.** `src/models/evaluate.py` goes past a single AUC number into calibration — whether a predicted 0.7 really means "70% of these cancel" — and cost-sensitive threshold selection, since a missed cancellation and a false alarm don't cost the same thing to the business. `src/models/explain.py` computes exact SHAP values via `TreeExplainer` on the final model, both globally (which features matter overall) and locally (why *this* booking scored the way it did) — and the top global features it independently discovers are the same ones four other methods surfaced earlier in the pipeline, which is the kind of convergent evidence that's hard to fake.

### Production layer
**Serving.** `src/serving/inference.py` and `app.py` wrap the model in a FastAPI service with Pydantic-validated input, a model loaded once at startup (not per-request), a `/health` readiness probe, and a Docker image built with layer caching and a non-root user. The design principle is training-serving skew prevention: the serving path imports the actual training-time feature function rather than re-implementing it.

**Streaming.** `src/serving/stream_producer.py` and `stream_consumer.py` simulate a Kafka-shaped producer/consumer pair against the same `InferencePipeline` — a plain Python generator standing in for a broker, deliberately, so swapping in real Kafka later touches only the producer, never the scoring logic. It measures its own p50/p95 latency and writes a JSON-Lines prediction log built specifically to feed the next stage.

**Monitoring.** `src/monitoring/drift.py` implements PSI and Kolmogorov–Smirnov drift detection from first principles against the training distribution. `monitor.py` layers a performance-decay policy on top, keyed off a *relative* drop from baseline rather than a fixed floor, because a monitoring threshold calibrated for a different model's baseline is worse than no threshold. `retrain_flow.py` orchestrates the whole check as a Prefect flow with automatic retries and a `.serve(cron=...)` path to real scheduling — currently stubbing the actual retraining call so the flow is safe to demo repeatedly without a 15-minute training run in the loop.

### Business layer
**Demand + pricing.** `src/models/demand.py` aggregates individual bookings into a monthly time series and forecasts it with lag/rolling/calendar features, validated with a proper walk-forward backtest rather than a random split — because there is exactly one valid direction to evaluate a forecaster in. `src/models/pricing.py` combines that forecast with a price-elasticity curve to recommend a revenue-maximizing rate, deliberately via an inspectable grid sweep rather than a black-box optimizer, so a revenue manager can see the whole curve and understand why a price was chosen — framed explicitly as a decision-support tool, not an autonomous price-setter.

## Quickstart

```bash
git clone https://github.com/surajbhat16/Hotel_ML.git
cd Hotel_ML

uv sync --all-extras --dev
uv run python src/ingestion/generate_dataset.py

uv run python src/processing/clean.py
uv run python src/processing/split.py
uv run python src/features/engineer.py
uv run python src/features/encode.py
uv run python src/features/scale.py
uv run python src/models/train.py --quick   # ~1 min; drop --quick for the full Optuna sweep

uv run python src/models/demand.py     # writes data/processed/demand_series.csv, needed by /forecast and /price

uv run uvicorn src.serving.app:app --reload
# POST a booking to http://127.0.0.1:8000/predict     — cancellation risk
# GET  http://127.0.0.1:8000/forecast                 — next-period demand
# POST http://127.0.0.1:8000/price                    — revenue-maximising nightly rate
# interactive docs (try all three from the browser) at /docs
```

Run the whole quality gate:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

Or build and run the container directly:

```bash
docker build -t hotel-cancellation-api .
docker run -p 8000:8000 hotel-cancellation-api
```

## Project layout

```
hotel-ml/
├── data/
│   ├── raw/            # immutable source data, regenerated, never edited by hand
│   ├── interim/        # structurally cleaned, reproducible from raw
│   └── processed/      # split, engineered, encoded, scaled — model-ready
├── src/
│   ├── ingestion/      # synthetic dataset generation
│   ├── processing/     # cleaning, split, EDA + leakage hunting
│   ├── features/       # engineering, encoding, scaling, selection
│   ├── models/         # CV, imbalance, training, evaluation, explainability,
│   │                   # demand forecasting, pricing
│   ├── serving/        # FastAPI app, inference pipeline, streaming sim
│   └── monitoring/     # drift detection, performance policy, Prefect flow
├── artifacts/           # trained model, encoders, scaler stats, evaluation/SHAP reports
├── tests/                # 92 tests across 10 files — unit, leakage-guard, and integration
├── notebooks/            # presentation-grade EDA (regenerable from a script, not hand-edited)
├── Dockerfile
└── .github/workflows/    # CI: ruff, mypy, pytest on every push and PR
```

## Engineering practices this repo actually follows

- **Leakage prevention is structural, not a checklist.** The train/test split happens before any statistic is computed. Every fitted transform (imputation mode, encoding map, scaler center/scale) is learned on train and replayed onto test and production — never re-derived from data it shouldn't see. The riskiest single step, target encoding, is defended with out-of-fold computation and has a dedicated test proving a row's encoding never depends on its own label.
- **Time series is treated like time series.** Cross-validation, the train/test split, and the demand-forecast backtest all walk forward in time. None of them shuffle.
- **Baselines come before complexity, always.** Every model — cancellation and demand — is compared against a deliberately simple baseline before any tuning effort is trusted, and the baseline result is reported even when it's a wash.
- **Calibration is checked, not assumed.** A model can rank perfectly and still report untrustworthy probabilities; this one is tested for both, separately, because the pricing layer depends on the actual numeric probability, not just its rank.
- **Explainability runs against the real, final model.** SHAP values here are computed on the tuned production model with `TreeExplainer`, not approximated from a proxy.
- **The serving path reuses training code by import, not by copy.** This is the actual mechanism that prevents training-serving skew — not a comment promising it.
- **Every finding, including the negative ones, is logged as a decision with a reason.** The retraining policy returns human-readable reasons alongside its verdict; the SMOTE leakage trap is demonstrated empirically, not just described.

## Known limitations & what's next

I'd rather list these than have someone else find them first.

1. **Target encoding is a no-op in production.** `encode.py` persists per-level maps for frequency encoding but only summary statistics (prior, smoothing constant) for target encoding — so the serving path currently falls back to a constant prior for `country_te`/`agent_te`/`company_te` on every request, regardless of the actual value submitted. SHAP ranked `agent_te` as a top-12 global feature on the trained model; the served version is quietly not using it. Fix is straightforward: persist the full per-level map the way frequency encoding already does.
2. **The default price-elasticity assumption produces boundary solutions, not interior ones.** With the current constant-elasticity demand curve, the revenue-maximizing price is mathematically guaranteed to land on the edge of the search grid for any elasticity with magnitude greater than 1 — meaning the "optimal" price is really just the cheapest or most expensive price the grid allows to be considered, not a genuine sweet spot. The API now surfaces this directly as `optimum_at_price_bound` in the `/price` response instead of burying it. Needs either a real cost/capacity floor or a saturating demand curve to produce an interior optimum.
3. **Single-row inference isn't batched.** The monitoring flow's performance check scores a 4,000-row window one booking at a time through the full single-row pipeline, which takes roughly 2.5 minutes — fine today, a real bottleneck if the monitoring window or schedule frequency grows. Needs a batched `transform`/`predict` path. The same limitation is why `/price`'s portfolio cancellation-rate estimate is computed once at API startup (sampling 200 bookings, ~5s) rather than per request — see [docs/PhaseAPI.pdf](docs/PhaseAPI.pdf).
4. **CI doesn't yet provision a model artifact.** The serving integration tests load the real trained model from `artifacts/`, which is git-ignored by design (models don't belong in git history) — so CI will fail on a clean checkout until a training or artifact-restore step is added ahead of the test job.
5. **Cancellation model complexity isn't earning its keep yet.** Optuna-tuned LightGBM ties a plain logistic regression on PR-AUC. That's an honest result, not a bug, but the real next step is feature work — richer agent/company signal, better temporal features — rather than more hyperparameter search, which has already shown diminishing returns here.

None of these are hidden — they're the actual state of the repo, and fixing #1 and #4 is next on my list.

~~The pricing engine doesn't call the cancellation model.~~ **Fixed.** `/price` now calls `estimate_base_cancel_rate()` in `pricing.py`, which scores a real sample of bookings through the trained `InferencePipeline` and averages the result — replacing the hardcoded 0.33 constant everywhere pricing runs, including the standalone `pricing.py` script. Full details in [docs/PhaseAPI.pdf](docs/PhaseAPI.pdf).

## Tech stack

| Layer | Tools |
|---|---|
| Data / features | pandas, numpy, scikit-learn |
| Modeling | LightGBM, Optuna, imbalanced-learn |
| Tracking | MLflow (local, SQLite-backed) |
| Explainability | SHAP |
| Serving | FastAPI, Pydantic, uvicorn, Docker |
| Orchestration | Prefect 3 |
| Packaging | uv (dependency resolution + lockfile), hatchling |
| Quality gates | ruff, mypy (strict), pytest, pre-commit, GitHub Actions |

## Running the tests

```bash
uv run pytest              # 92 tests
uv run mypy src             # strict, zero untyped defs allowed in src/
uv run ruff check . && uv run ruff format --check .
```

---

Built solo as a full-scope demonstration of what a production ML system's *edges* actually look like — not just the model, but the split, the leakage guards, the calibration check, the serving path that doesn't drift from training, and the honest list of what's still rough.
