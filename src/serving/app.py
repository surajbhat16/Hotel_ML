"""
Stage 8b — FastAPI scoring service.

A production-shaped real-time API around the cancellation model. Design choices
that matter in an interview:

  - PYDANTIC request validation. The BookingRequest schema rejects malformed input
    at the edge (wrong types, out-of-range values) before it ever reaches the
    model, returning a clear 422 instead of a 500. Validation is part of the
    contract, not an afterthought.

  - MODEL LOADED ONCE at startup via lifespan, not per request. Loading the
    booster and artifacts on every call would add hundreds of ms of latency.

  - SEPARATION of transport (this file) from logic (inference.py). The API is a
    thin shell; all feature/scoring logic lives in the reusable pipeline, which
    is what prevents training-serving skew.

  - HEALTH endpoint for readiness checks (load balancers, k8s liveness probes).

Run locally:
    uvicorn src.serving.app:app --reload
    # then POST to http://127.0.0.1:8000/predict  (docs at /docs)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from src.serving.inference import InferencePipeline
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.serving.inference import InferencePipeline

try:
    from src.models.demand import forecast_next
    from src.models.pricing import estimate_base_cancel_rate, price_for_period
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.demand import forecast_next
    from src.models.pricing import estimate_base_cancel_rate, price_for_period

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("api")

# Default operating threshold: the cost-optimal value found in Stage 7.
DEFAULT_THRESHOLD = 0.30

# Reference nightly rate for pricing when the caller doesn't supply one.
DEFAULT_REF_PRICE = 110.0

DEMAND_SERIES_FILE = "data/processed/demand_series.csv"
BOOKING_SAMPLE_FILE = "data/processed/test.csv"
# How many real bookings the portfolio cancellation-rate estimate is averaged
# over. Computed once at startup (see lifespan) -- not per request, for the
# same reason the model itself is loaded once: re-scoring hundreds of rows on
# every /price call would add seconds of latency for no benefit, since the
# portfolio's risk profile doesn't change between requests.
CANCEL_RATE_SAMPLE_SIZE = 200

# Module-level holder; populated at startup so the model loads exactly once.
_state: dict[str, Any] = {}


class BookingRequest(BaseModel):
    """The raw booking contract. Field constraints reject bad input at the edge."""

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0, le=1000, description="days between booking and arrival")
    arrival_date_year: int = Field(ge=2010, le=2035)
    arrival_date_month: str
    adults: int = Field(ge=0, le=20)
    children: int = Field(ge=0, le=20)
    babies: int = Field(ge=0, le=20)
    market_segment: str
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: str
    adr: float = Field(ge=0, le=10000, description="average daily rate")
    previous_cancellations: int = Field(ge=0)
    booking_changes: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
    is_repeated_guest: int = Field(ge=0, le=1)
    country: str = "PRT"
    agent: float = 0.0
    company: float = 0.0
    agent_missing: int = Field(default=0, ge=0, le=1)
    company_missing: int = Field(default=1, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "hotel": "City Hotel",
                "lead_time": 200,
                "arrival_date_year": 2017,
                "arrival_date_month": "August",
                "adults": 2,
                "children": 0,
                "babies": 0,
                "market_segment": "Groups",
                "deposit_type": "Non Refund",
                "customer_type": "Transient",
                "adr": 120.0,
                "previous_cancellations": 1,
                "booking_changes": 0,
                "total_of_special_requests": 0,
                "is_repeated_guest": 0,
                "country": "PRT",
                "agent": 9.0,
                "company": 0.0,
                "agent_missing": 0,
                "company_missing": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    cancellation_probability: float
    will_cancel: bool
    threshold: float
    risk_band: Literal["low", "medium", "high"]


class ForecastResponse(BaseModel):
    """Next period's forecast net demand (arrivals net of cancellations)."""

    forecast_bookings: float
    based_on_periods: int


class PriceRequest(BaseModel):
    reference_price: float = Field(default=DEFAULT_REF_PRICE, gt=0)
    elasticity: float = Field(
        default=-1.2, lt=0, description="demand % change per 1% price change; must be negative"
    )


class PriceResponse(BaseModel):
    recommended_price: float
    reference_price: float
    forecast_demand: float
    expected_revenue_at_recommended: float
    expected_revenue_at_reference: float
    revenue_uplift_pct: float
    base_cancel_rate: float
    optimum_at_price_bound: bool


_DEFAULT_PRICE_REQUEST = PriceRequest()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model + artifacts once when the service starts."""
    log.info("Loading inference pipeline...")
    pipeline = InferencePipeline()
    _state["pipeline"] = pipeline

    log.info("Loading demand series...")
    _state["demand_series"] = pd.read_csv(DEMAND_SERIES_FILE)

    log.info(
        "Estimating portfolio cancellation rate from %d sampled bookings via the "
        "trained model (this replaces pricing's old hardcoded constant)...",
        CANCEL_RATE_SAMPLE_SIZE,
    )
    booking_sample = pd.read_csv(BOOKING_SAMPLE_FILE).drop(columns=["is_canceled"])
    _state["base_cancel_rate"] = estimate_base_cancel_rate(
        pipeline, booking_sample, sample_size=CANCEL_RATE_SAMPLE_SIZE
    )
    log.info("Portfolio cancellation rate estimate: %.4f", _state["base_cancel_rate"])

    log.info("Service ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Hotel Cancellation Scoring API",
    version="1.0.0",
    description="Real-time cancellation-risk scoring for hotel bookings.",
    lifespan=lifespan,
)


def _risk_band(p: float) -> str:
    if p < 0.25:
        return "low"
    if p < 0.6:
        return "medium"
    return "high"


@app.get("/health")
def health() -> dict:
    """Readiness probe: confirms the model is loaded and servable."""
    ready = "pipeline" in _state
    return {"status": "ok" if ready else "loading", "model_loaded": ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest, threshold: float = DEFAULT_THRESHOLD) -> PredictionResponse:
    """Score a single booking's cancellation risk."""
    pipeline = _state["pipeline"]
    result = pipeline.predict(booking.model_dump(), threshold=threshold)
    return PredictionResponse(
        cancellation_probability=result["cancellation_probability"],
        will_cancel=result["will_cancel"],
        threshold=result["threshold"],
        risk_band=_risk_band(result["cancellation_probability"]),  # type: ignore[arg-type]
    )


@app.get("/forecast", response_model=ForecastResponse)
def forecast() -> ForecastResponse:
    """Forecast next period's net bookings (Stage 11 demand model)."""
    series = _state["demand_series"]
    prediction = forecast_next(series)
    return ForecastResponse(
        forecast_bookings=round(prediction, 1),
        based_on_periods=len(series),
    )


@app.post("/price", response_model=PriceResponse)
def price(request: PriceRequest = _DEFAULT_PRICE_REQUEST) -> PriceResponse:
    """Recommend a revenue-maximising nightly rate for the next period.

    Ties the demand forecast (Stage 11) to the cancellation model's live
    portfolio-risk estimate (Stage 1-7), computed once at startup -- see
    lifespan. This is a decision-support number, not an autonomous price-setter.
    """
    series = _state["demand_series"]
    base_cancel = _state["base_cancel_rate"]
    result = price_for_period(
        series,
        ref_price=request.reference_price,
        base_cancel=base_cancel,
        elasticity=request.elasticity,
    )
    return PriceResponse(
        recommended_price=result["recommended_price"],
        reference_price=result["reference_price"],
        forecast_demand=result["forecast_demand"],
        expected_revenue_at_recommended=result["expected_revenue_at_recommended"],
        expected_revenue_at_reference=result["expected_revenue_at_reference"],
        revenue_uplift_pct=result["revenue_uplift_pct"],
        base_cancel_rate=round(base_cancel, 4),
        optimum_at_price_bound=result["optimum_at_price_bound"],
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "hotel-cancellation",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/forecast", "/price"],
    }
