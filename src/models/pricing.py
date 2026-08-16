"""
Stage 12 — Dynamic pricing optimization.

The capstone that ties the platform together. Pricing is where the two models
meet: the demand forecast (Stage 11) tells us how many bookings to expect at a
given price, and the cancellation model (Stages 1-7) tells us how likely each
booking is to actually stick. We choose the price that maximises EXPECTED REVENUE.

The core idea — expected revenue at price p:

    expected_revenue(p) = demand(p) * p * (1 - cancel_prob(p))

  - demand(p): higher price -> fewer bookings. We model this with a simple,
    transparent price-elasticity curve anchored on the forecast at a reference
    price. (A production system would estimate elasticity from historical
    price/demand pairs; we make the assumption explicit and adjustable.)
  - p: the nightly rate (ADR).
  - (1 - cancel_prob(p)): higher prices and non-refundable terms shift
    cancellation risk, so realised revenue must discount for expected
    cancellations.

We sweep a grid of candidate prices and pick the revenue-maximising one. This is
deliberately interpretable rather than a black-box optimiser: a revenue manager
can see the whole curve and understand why a price was chosen.

IMPORTANT FRAMING: this is a decision-support tool, not an autonomous price-setter.
It recommends; a human sets guardrails (floors, ceilings, fairness constraints).

Run:  python src/models/pricing.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("pricing")

SERIES_IN = "data/processed/demand_series.csv"


def demand_at_price(
    base_demand: float, price: float, ref_price: float, elasticity: float = -1.2
) -> float:
    """Constant-elasticity demand curve: demand scales with (price/ref)^elasticity.

    elasticity < 0 means higher price -> lower demand. -1.2 is a plausible,
    slightly elastic default for hospitality (a 1% price rise -> ~1.2% demand
    drop). This is the key modelling assumption and is exposed as a parameter.
    """
    return float(base_demand * (price / ref_price) ** elasticity)


def cancel_prob_at_price(
    base_cancel: float, price: float, ref_price: float, sensitivity: float = 0.15
) -> float:
    """Cancellation probability rises modestly with price (pricier bookings are a
    bit likelier to be reconsidered), capped to [0, 0.95]. sensitivity controls
    how strongly price nudges cancellation."""
    adjusted = base_cancel * (1 + sensitivity * (price / ref_price - 1))
    return float(np.clip(adjusted, 0.0, 0.95))


def expected_revenue(
    price: float,
    base_demand: float,
    ref_price: float,
    base_cancel: float,
    elasticity: float = -1.2,
) -> float:
    """Expected realised revenue at a candidate price."""
    d = demand_at_price(base_demand, price, ref_price, elasticity)
    c = cancel_prob_at_price(base_cancel, price, ref_price)
    return float(d * price * (1 - c))


def optimize_price(
    base_demand: float,
    ref_price: float,
    base_cancel: float,
    elasticity: float = -1.2,
    price_min: float | None = None,
    price_max: float | None = None,
    n_grid: int = 100,
) -> dict:
    """Sweep a price grid and return the revenue-maximising price plus the full
    curve for inspection."""
    price_min = price_min or ref_price * 0.6
    price_max = price_max or ref_price * 1.8
    grid = np.linspace(price_min, price_max, n_grid)

    revenues = np.array(
        [expected_revenue(p, base_demand, ref_price, base_cancel, elasticity) for p in grid]
    )
    best_i = int(np.argmax(revenues))
    best_price = float(grid[best_i])

    ref_rev = expected_revenue(ref_price, base_demand, ref_price, base_cancel, elasticity)
    best_rev = float(revenues[best_i])

    # Flag when the optimum sits on a grid boundary — it signals the true optimum
    # may lie outside the allowed range, i.e. a price guardrail is binding.
    at_bound = best_i == 0 or best_i == len(grid) - 1

    return {
        "recommended_price": round(best_price, 2),
        "reference_price": round(ref_price, 2),
        "expected_revenue_at_recommended": round(best_rev, 2),
        "expected_revenue_at_reference": round(ref_rev, 2),
        "revenue_uplift_pct": round(100 * (best_rev - ref_rev) / max(ref_rev, 1e-9), 2),
        "optimum_at_price_bound": at_bound,
        "price_grid": grid.round(2).tolist(),
        "revenue_curve": revenues.round(2).tolist(),
        "elasticity": elasticity,
    }


def price_for_period(
    demand_series: pd.DataFrame,
    ref_price: float,
    base_cancel: float,
    elasticity: float = -1.2,
) -> dict:
    """End-to-end: forecast next-period demand, then optimise price for it.
    This is the function that ties Stage 11 (demand) and the cancellation model
    (base_cancel) into a single pricing decision."""
    try:
        from src.models.demand import forecast_next
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.models.demand import forecast_next

    forecast = forecast_next(demand_series)
    log.info("Forecast next-period demand: %.0f bookings", forecast)
    result = optimize_price(forecast, ref_price, base_cancel, elasticity)
    result["forecast_demand"] = round(forecast, 1)
    return result


if __name__ == "__main__":
    series = pd.read_csv(SERIES_IN)

    # Reference price ~ typical ADR; base cancel ~ portfolio rate.
    REF_PRICE = 110.0
    BASE_CANCEL = 0.33

    log.info("=" * 60)
    log.info("Single-price optimisation demo (fixed demand):")
    single = optimize_price(base_demand=3000, ref_price=REF_PRICE, base_cancel=BASE_CANCEL)
    log.info(
        "  Reference price:   $%.2f -> expected revenue $%.0f",
        single["reference_price"],
        single["expected_revenue_at_reference"],
    )
    log.info(
        "  Recommended price: $%.2f -> expected revenue $%.0f",
        single["recommended_price"],
        single["expected_revenue_at_recommended"],
    )
    log.info("  Revenue uplift:    %+.2f%%", single["revenue_uplift_pct"])

    log.info("=" * 60)
    log.info("End-to-end: forecast demand -> optimise price")
    end2end = price_for_period(series, ref_price=REF_PRICE, base_cancel=BASE_CANCEL)
    log.info("  Forecast demand:   %.0f", end2end["forecast_demand"])
    log.info(
        "  Recommended price: $%.2f (uplift %+.2f%% vs reference)",
        end2end["recommended_price"],
        end2end["revenue_uplift_pct"],
    )

    log.info("=" * 60)
    log.info("Elasticity sensitivity (how the recommendation shifts):")
    for e in (-0.8, -1.2, -2.0):
        r = optimize_price(3000, REF_PRICE, BASE_CANCEL, elasticity=e)
        log.info(
            "  elasticity %.1f -> recommend $%.2f (uplift %+.1f%%)",
            e,
            r["recommended_price"],
            r["revenue_uplift_pct"],
        )
