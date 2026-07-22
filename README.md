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
- [ ] Stage 1 — Data cleaning
- [ ] Stage 2 — EDA + leakage hunting
- [ ] Stage 3 — Feature engineering + encoding
- [ ] Stage 4 — Feature selection + scaling + dimensionality reduction
- [ ] Stage 5 — Imbalance handling + cross-validation design
- [ ] Stage 6 — Model training + hyperparameter tuning
- [ ] Stage 7 — Evaluation + calibration + explainability
- [ ] Stage 8 — Serving (FastAPI + Docker)
- [ ] Stage 9 — Streaming simulation
- [ ] Stage 10 — Drift detection + monitoring + retraining
- [ ] Stage 11 — Demand forecasting model
- [ ] Stage 12 — Dynamic pricing optimization

## Setup

```bash
pip install -r requirements.txt
python src/ingestion/generate_dataset.py   # creates data/raw/hotel_bookings.csv
```
