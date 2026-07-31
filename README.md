# Hotel Booking ML Platform

End-to-end, production-shaped ML system built locally with open-source tools.
Three interlocking problems on hotel-booking data:

1. **Cancellation prediction** — will a booking be canceled? (classification)
2. **Demand forecasting** — how many bookings next period? (time series)
3. **Dynamic pricing** — what price maximizes expected revenue? (optimization)

## Project layout

```
hotel-ml/
├── data/
│   ├── raw/          # immutable source data (never edited)
│   ├── interim/      # partially processed, reproducible from raw
│   └── processed/    # final model-ready tables
├── src/
│   ├── ingestion/    # dataset generation + streaming simulation
│   ├── processing/   # cleaning, EDA, splitting
│   ├── features/     # feature engineering, selection, scaling
│   ├── models/       # training, tuning, evaluation
│   ├── serving/      # FastAPI scoring service
│   └── monitoring/   # drift detection + retraining triggers
├── configs/          # YAML config (paths, params, thresholds)
├── artifacts/        # trained models, encoders, reports
├── notebooks/        # exploratory work
└── tests/            # unit tests
```

## Pipeline stages (roadmap)

- [x] Stage 0 — Scaffold + data acquisition
- [x] Stage 1 — Data cleaning
- [x] Stage 2 — EDA + leakage hunting
- [x] Stage 3 — Feature engineering + encoding
- [x] Stage 4 — Feature selection + scaling + dimensionality reduction
- [x] Stage 5 — Imbalance handling + cross-validation design
- [x] Stage 6 — Model training + hyperparameter tuning
- [x] Stage 7 — Evaluation + calibration + explainability
- [x] Stage 8 — Serving (FastAPI + Docker)
- [x] Stage 9 — Streaming simulation
- [ ] Stage 10 — Drift detection + monitoring + retraining
- [ ] Stage 11 — Demand forecasting model
- [ ] Stage 12 — Dynamic pricing optimization

## Exploratory analysis

A comprehensive, presentation-grade EDA lives in `notebooks/01_eda.ipynb` (reads the **train split only** to respect the leakage firewall). To run it:

```bash
uv sync --extra notebook --dev
uv run jupyter lab notebooks/01_eda.ipynb
```

It's regenerable from `notebooks/build_eda_notebook.py`.

## Setup

```bash
uv sync --all-extras --dev
uv run python src/ingestion/generate_dataset.py   # creates data/raw/hotel_bookings.csv
```
