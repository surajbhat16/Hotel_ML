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
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from src.serving.inference import InferencePipeline
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.serving.inference import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("api")

# Default operating threshold: the cost-optimal value found in Stage 7.
DEFAULT_THRESHOLD = 0.30

# Module-level holder; populated at startup so the model loads exactly once.
_state: dict[str, InferencePipeline] = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model + artifacts once when the service starts."""
    log.info("Loading inference pipeline...")
    _state["pipeline"] = InferencePipeline()
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


@app.get("/")
def root() -> dict:
    return {"service": "hotel-cancellation", "docs": "/docs", "health": "/health"}
