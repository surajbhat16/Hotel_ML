"""Tests for Stage 8 serving. Run: pytest tests/ -q

These exercise the anti-skew inference pipeline and the FastAPI endpoints against
the real trained artifacts, so they double as an integration test of the whole
train -> serve path.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.inference import InferencePipeline

VALID_BOOKING = {
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


# ------------------------------ inference pipeline ------------------------------


@pytest.fixture(scope="module")
def pipe():
    return InferencePipeline()


def test_transform_produces_expected_columns(pipe):
    x = pipe.transform(VALID_BOOKING)
    assert list(x.columns) == pipe.expected_columns
    assert len(x) == 1


def test_transform_all_numeric(pipe):
    x = pipe.transform(VALID_BOOKING)
    import numpy as np

    assert x.select_dtypes(exclude=[np.number, bool]).shape[1] == 0


def test_probability_in_unit_interval(pipe):
    p = pipe.predict_proba(VALID_BOOKING)
    assert 0.0 <= p <= 1.0


def test_high_risk_scores_above_low_risk(pipe):
    """Sanity: a risky booking must score higher than a safe one — proves the
    serving transform preserves the model's learned signal."""
    low = dict(
        VALID_BOOKING,
        lead_time=3,
        deposit_type="No Deposit",
        market_segment="Direct",
        total_of_special_requests=3,
        is_repeated_guest=1,
        previous_cancellations=0,
    )
    assert pipe.predict_proba(VALID_BOOKING) > pipe.predict_proba(low)


def test_unseen_category_does_not_crash(pipe):
    """A country never seen in training must fall back gracefully, not error."""
    booking = dict(VALID_BOOKING, country="ZZZ")
    p = pipe.predict_proba(booking)
    assert 0.0 <= p <= 1.0


# ----------------------------------- API ---------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_payload(client):
    r = client.post("/predict", json=VALID_BOOKING)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["cancellation_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert isinstance(body["will_cancel"], bool)


def test_predict_rejects_bad_enum(client):
    bad = dict(VALID_BOOKING, deposit_type="Bogus")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_lead_time(client):
    bad = dict(VALID_BOOKING, lead_time=-5)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_BOOKING.items() if k != "adr"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_custom_threshold_changes_label(client):
    # At a very high threshold even a risky booking should flip to will_cancel=False.
    r_hi = client.post("/predict?threshold=0.99", json=VALID_BOOKING)
    r_lo = client.post("/predict?threshold=0.01", json=VALID_BOOKING)
    assert r_hi.json()["will_cancel"] is False
    assert r_lo.json()["will_cancel"] is True
