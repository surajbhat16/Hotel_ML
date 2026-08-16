"""
Stage 11 — Demand forecasting.

The second ML problem in the platform. Cancellation prediction is per-booking;
demand forecasting is aggregate: how many bookings (net of cancellations) will
arrive per period? The hotel needs this to plan staffing, inventory, and — feeding
Stage 12 — pricing.

Time-series modelling essentials shown here:

  1. AGGREGATION: collapse individual bookings into a regular time series
     (arrivals per month), the unit we actually forecast.

  2. BASELINES FIRST (again): a seasonal-naive forecast (this month ~ same month
     last year) is a genuinely hard baseline to beat in seasonal data. If a fancy
     model can't beat it, use the baseline.

  3. LAG & CALENDAR FEATURES: a tree model can forecast if we hand it the recent
     past (lags), rolling means, and calendar signals (month, cyclical encoding).
     This turns forecasting into supervised regression.

  4. BACKTESTING with a rolling origin: never evaluate a forecaster on random
     folds. We walk forward through time, always predicting the future from the
     past, exactly as in production.

Metrics: MAE and MAPE (interpretable, scale-aware) rather than RMSE alone.

Run:  python src/models/demand.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("demand")

CLEAN_IN = "data/interim/hotel_clean.csv"
SERIES_OUT = "data/processed/demand_series.csv"

MONTH_NUM = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def build_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bookings into arrivals-per-month, net of cancellations.

    'Realised demand' = bookings that were NOT cancelled, since that's what
    actually consumes rooms. We also keep gross bookings for context.
    """
    d = df.copy()
    d["month_num"] = d["arrival_date_month"].map(MONTH_NUM)
    d["period"] = d["arrival_date_year"] * 100 + d["month_num"]

    grouped = d.groupby("period").agg(
        gross_bookings=("is_canceled", "size"),
        cancellations=("is_canceled", "sum"),
    )
    grouped["realised_demand"] = grouped["gross_bookings"] - grouped["cancellations"]
    grouped = grouped.sort_index().reset_index()
    grouped["year"] = grouped["period"] // 100
    grouped["month"] = grouped["period"] % 100
    log.info(
        "Built demand series: %d monthly periods (%d..%d)",
        len(grouped),
        grouped["period"].min(),
        grouped["period"].max(),
    )
    return grouped


def add_time_features(series: pd.DataFrame, target: str = "realised_demand") -> pd.DataFrame:
    """Lag, rolling, and calendar features that let a regressor forecast."""
    s = series.copy()
    for lag in (1, 2, 3, 12):
        s[f"lag_{lag}"] = s[target].shift(lag)
    s["roll_mean_3"] = s[target].shift(1).rolling(3).mean()
    s["roll_mean_6"] = s[target].shift(1).rolling(6).mean()
    s["month_sin"] = np.sin(2 * np.pi * s["month"] / 12)
    s["month_cos"] = np.cos(2 * np.pi * s["month"] / 12)
    return s


def seasonal_naive_forecast(series: pd.DataFrame, target: str = "realised_demand") -> pd.Series:
    """Baseline: predict each month as the same month one year earlier (lag-12).
    In strongly seasonal data this is a formidable baseline."""
    return series[target].shift(12)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def backtest_lgbm(
    series: pd.DataFrame, target: str = "realised_demand", min_train: int = 18
) -> dict:
    """Rolling-origin backtest: for each step from min_train onward, train on all
    prior months and predict the next one. Compares LightGBM vs seasonal-naive."""
    from lightgbm import LGBMRegressor

    feat = add_time_features(series, target).dropna().reset_index(drop=True)
    feature_cols = [
        c for c in feat.columns if c.startswith(("lag_", "roll_", "month_sin", "month_cos"))
    ]

    lgbm_preds, naive_preds, actuals = [], [], []
    for i in range(min_train, len(feat)):
        train = feat.iloc[:i]
        test = feat.iloc[i : i + 1]
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
        model.fit(train[feature_cols], train[target])
        lgbm_preds.append(float(model.predict(test[feature_cols])[0]))
        naive_preds.append(float(test["lag_12"].iloc[0]))
        actuals.append(float(test[target].iloc[0]))

    actuals_a = np.array(actuals)
    results = {
        "n_test_points": len(actuals),
        "lgbm_mae": round(mae(actuals_a, np.array(lgbm_preds)), 2),
        "lgbm_mape": round(mape(actuals_a, np.array(lgbm_preds)), 2),
        "naive_mae": round(mae(actuals_a, np.array(naive_preds)), 2),
        "naive_mape": round(mape(actuals_a, np.array(naive_preds)), 2),
    }
    results["lgbm_beats_naive"] = results["lgbm_mae"] < results["naive_mae"]
    return results


def forecast_next(series: pd.DataFrame, target: str = "realised_demand") -> float:
    """Fit on all available history and forecast the next month's demand.
    Used by the pricing stage."""
    from lightgbm import LGBMRegressor

    feat = add_time_features(series, target)
    train = feat.dropna()
    feature_cols = [
        c for c in feat.columns if c.startswith(("lag_", "roll_", "month_sin", "month_cos"))
    ]
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[feature_cols], train[target])
    # Build the feature row for the next period from the tail of history.
    next_row = feat.iloc[[-1]][feature_cols]
    return float(model.predict(next_row)[0])


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_IN)
    series = build_demand_series(df)
    series.to_csv(SERIES_OUT, index=False)
    log.info("Wrote %s", SERIES_OUT)
    log.info("=" * 60)
    log.info(
        "Demand series (realised demand per month):\n%s",
        series[["period", "gross_bookings", "cancellations", "realised_demand"]].to_string(
            index=False
        ),
    )

    log.info("=" * 60)
    log.info("Backtesting LightGBM vs seasonal-naive (rolling origin)...")
    results = backtest_lgbm(series)
    log.info("  LightGBM   MAE=%.1f  MAPE=%.1f%%", results["lgbm_mae"], results["lgbm_mape"])
    log.info("  Seasonal-naive MAE=%.1f  MAPE=%.1f%%", results["naive_mae"], results["naive_mape"])
    log.info("  LightGBM beats naive: %s", results["lgbm_beats_naive"])
