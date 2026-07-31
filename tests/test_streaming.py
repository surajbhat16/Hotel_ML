"""Tests for Stage 9 streaming. Run: pytest tests/ -q"""

import pytest

from src.serving.stream_consumer import StreamingScorer
from src.serving.stream_producer import stream_bookings

# --------------------------------- producer ---------------------------------


def test_producer_yields_requested_count():
    events = list(stream_bookings(limit=10, shuffle=True))
    assert len(events) == 10


def test_producer_event_shape():
    event = next(iter(stream_bookings(limit=1, shuffle=False)))
    assert "event_id" in event
    assert "booking" in event
    assert "_true_label" in event
    # The booking dict must carry the raw fields the model's serving path needs.
    for field in ["hotel", "lead_time", "deposit_type", "adr"]:
        assert field in event["booking"]


def test_producer_is_lazy_generator():
    """Must be a generator (streamed), not a materialised list."""
    import types

    gen = stream_bookings(limit=5)
    assert isinstance(gen, types.GeneratorType)


def test_producer_shuffle_changes_order():
    a = [e["booking"]["lead_time"] for e in stream_bookings(limit=20, shuffle=True, seed=1)]
    b = [e["booking"]["lead_time"] for e in stream_bookings(limit=20, shuffle=True, seed=2)]
    assert a != b  # different seeds -> different order


# --------------------------------- consumer ---------------------------------


@pytest.fixture(scope="module")
def scorer():
    return StreamingScorer(threshold=0.3, window=50)


def test_process_event_returns_scored_record(scorer):
    event = next(iter(stream_bookings(limit=1)))
    rec = scorer.process_event(event)
    assert 0.0 <= rec["probability"] <= 1.0
    assert rec["prediction"] in {0, 1}
    assert rec["risk_band"] in {"low", "medium", "high"}
    assert rec["latency_ms"] >= 0


def test_running_metrics_update_incrementally():
    s = StreamingScorer(threshold=0.3)
    for event in stream_bookings(limit=30):
        s.process_event(event)
    m = s.running_metrics()
    assert m["processed"] == 30
    assert sum(m["risk_distribution"].values()) == 30
    assert m["running_accuracy"] is not None


def test_sliding_window_bounded():
    s = StreamingScorer(threshold=0.3, window=10)
    for event in stream_bookings(limit=40):
        s.process_event(event)
    # The recent-window deque must not exceed its maxlen.
    assert len(s.recent) == 10


def test_consumer_resilient_to_bad_event():
    """A malformed event must be skipped, not fatal — the stream continues."""
    s = StreamingScorer(threshold=0.3)
    good = list(stream_bookings(limit=5))
    bad = {"event_id": 999, "booking": {"garbage": True}, "_true_label": 0}
    stream = iter([*good[:2], bad, *good[2:]])
    final = s.run(stream, report_every=100)
    # 5 good events scored; the bad one skipped.
    assert final["processed"] == 5


def test_consumer_decoupled_from_producer():
    """The consumer must accept ANY iterator of events, not just our producer —
    proving the source can be swapped for real Kafka later."""
    s = StreamingScorer(threshold=0.3)
    events = list(stream_bookings(limit=3))
    final = s.run(iter(events), report_every=100)
    assert final["processed"] == 3
