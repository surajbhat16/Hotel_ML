"""Tests for Stage 11 demand + Stage 12 pricing. Run: pytest tests/ -q"""

import numpy as np
import pandas as pd

from src.models import demand, pricing

# --------------------------------- demand ---------------------------------


def _toy_bookings(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = list(demand.MONTH_NUM.keys())
    return pd.DataFrame(
        {
            "arrival_date_year": rng.choice([2015, 2016, 2017], n),
            "arrival_date_month": rng.choice(months, n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


def test_build_demand_series_columns():
    s = demand.build_demand_series(_toy_bookings())
    for col in ["period", "gross_bookings", "cancellations", "realised_demand", "month"]:
        assert col in s.columns


def test_realised_demand_is_gross_minus_cancellations():
    s = demand.build_demand_series(_toy_bookings())
    assert (s["realised_demand"] == s["gross_bookings"] - s["cancellations"]).all()


def test_add_time_features_creates_lags():
    s = demand.build_demand_series(_toy_bookings())
    feat = demand.add_time_features(s)
    for col in ["lag_1", "lag_12", "roll_mean_3", "month_sin", "month_cos"]:
        assert col in feat.columns


def test_mae_and_mape_zero_for_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert demand.mae(y, y) == 0.0
    assert demand.mape(y, y) < 1e-6


def test_mae_positive_for_errors():
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    assert demand.mae(y, p) == 15.0


def test_seasonal_naive_is_lag12():
    s = demand.build_demand_series(_toy_bookings())
    naive = demand.seasonal_naive_forecast(s)
    # Where defined, the naive forecast equals the value 12 periods earlier.
    for i in range(12, len(s)):
        if not np.isnan(naive.iloc[i]):
            assert naive.iloc[i] == s["realised_demand"].iloc[i - 12]


# --------------------------------- pricing ---------------------------------


def test_demand_falls_as_price_rises():
    """Negative elasticity => higher price yields lower demand."""
    lo = pricing.demand_at_price(1000, price=90, ref_price=100, elasticity=-1.2)
    hi = pricing.demand_at_price(1000, price=120, ref_price=100, elasticity=-1.2)
    assert hi < lo


def test_demand_equals_base_at_reference_price():
    d = pricing.demand_at_price(1000, price=100, ref_price=100, elasticity=-1.2)
    assert abs(d - 1000) < 1e-6


def test_cancel_prob_bounded():
    # Even at an extreme price the probability stays within [0, 0.95].
    p = pricing.cancel_prob_at_price(0.9, price=1000, ref_price=100, sensitivity=0.5)
    assert 0.0 <= p <= 0.95


def test_expected_revenue_is_positive():
    r = pricing.expected_revenue(100, base_demand=1000, ref_price=100, base_cancel=0.3)
    assert r > 0


def test_optimize_price_returns_valid_recommendation():
    result = pricing.optimize_price(base_demand=3000, ref_price=110, base_cancel=0.33)
    assert result["price_grid"][0] <= result["recommended_price"] <= result["price_grid"][-1]
    assert "revenue_uplift_pct" in result
    assert "optimum_at_price_bound" in result


def test_optimize_price_recommendation_beats_or_matches_reference():
    """The optimiser must never recommend a price with LOWER expected revenue than
    the reference — at worst it returns the reference."""
    result = pricing.optimize_price(base_demand=3000, ref_price=110, base_cancel=0.33)
    assert (
        result["expected_revenue_at_recommended"] >= result["expected_revenue_at_reference"] - 1e-6
    )


def test_elasticity_changes_recommendation():
    """More elastic demand should push the recommended price differently than
    inelastic demand — proving elasticity actually drives the result."""
    inelastic = pricing.optimize_price(3000, 110, 0.33, elasticity=-0.8)
    elastic = pricing.optimize_price(3000, 110, 0.33, elasticity=-2.0)
    assert inelastic["recommended_price"] != elastic["recommended_price"]
