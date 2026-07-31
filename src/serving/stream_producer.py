"""
Stage 9a — Streaming producer (simulated event source).

A real system has a PRODUCER that publishes booking events to a broker (Kafka,
Kinesis, Pub/Sub) and one or more CONSUMERS that react. We simulate the producer
here: it replays historical bookings one at a time as a stream of events, with
optional pacing to imitate real arrival timing.

Why a generator, not a list: a stream is unbounded and consumed lazily. Modelling
it as a Python generator (yield one event at a time) mirrors how a real consumer
pulls from a topic — you never hold the whole stream in memory, and back-pressure
is natural (the consumer sets the pace by how fast it iterates).

Each emitted event is a plain dict: exactly the shape the serving API accepts, so
the streaming path and the request/response path score identical payloads through
identical code (no skew).

Run:  python src/serving/stream_producer.py     # prints a few sample events
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

log = logging.getLogger("producer")

TEST_DATA = "data/processed/test.csv"

# The raw fields a booking event carries (what the model's serving path expects).
EVENT_FIELDS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "adults",
    "children",
    "babies",
    "market_segment",
    "deposit_type",
    "customer_type",
    "adr",
    "previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "is_repeated_guest",
    "country",
    "agent",
    "company",
    "agent_missing",
    "company_missing",
]


def _row_to_event(row: pd.Series, event_id: int) -> dict:
    """Convert a dataframe row into a booking event dict. We keep the true label
    (is_canceled) alongside under a private key so the consumer can score its own
    accuracy in the simulation — in production the label arrives later, if ever."""
    event = {k: row[k] for k in EVENT_FIELDS if k in row}
    # Coerce numpy scalar types to plain Python for clean JSON-like dicts.
    event = {k: (v.item() if hasattr(v, "item") else v) for k, v in event.items()}
    return {
        "event_id": event_id,
        "booking": event,
        "_true_label": int(row["is_canceled"]) if "is_canceled" in row else None,
    }


def stream_bookings(
    path: str = TEST_DATA,
    limit: int | None = None,
    delay: float = 0.0,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[dict]:
    """Yield booking events one at a time.

    limit   : stop after N events (None = whole file).
    delay   : seconds to sleep between events (imitates real arrival pacing).
    shuffle : randomise order so the stream isn't in stored order.
    """
    df = pd.read_csv(path)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)

    log.info("Producer streaming %d booking events from %s", len(df), Path(path).name)
    for i, (_, row) in enumerate(df.iterrows()):
        if delay > 0:
            time.sleep(delay)
        yield _row_to_event(row, event_id=i)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # Print a few sample events to show the stream shape.
    for event in stream_bookings(limit=3, shuffle=True):
        log.info(
            "event %d | label=%s | %s ...",
            event["event_id"],
            event["_true_label"],
            {k: event["booking"][k] for k in ["hotel", "lead_time", "deposit_type"]},
        )
