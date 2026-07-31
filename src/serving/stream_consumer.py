"""
Stage 9b — Streaming consumer (real-time scorer).

Pulls booking events from the producer, scores each one the instant it arrives
through the SAME InferencePipeline the REST API uses (Stage 8), and maintains
running metrics as the stream flows. This is the online-inference counterpart to
batch scoring.

What a real consumer must handle that a batch job doesn't:
  - ONE-AT-A-TIME latency: each event is scored on arrival, so per-event latency
    matters. We time it.
  - RUNNING STATE: metrics update incrementally (throughput, rolling accuracy,
    risk-band counts) rather than being computed once at the end.
  - RESILIENCE: a single malformed event must not kill the stream. We catch,
    log, and continue.
  - A SLIDING WINDOW of recent predictions, which is exactly what the drift
    monitor in Stage 10 will consume.

The consumer stays deliberately decoupled from the producer: it takes any iterator
of events, so the same code works against the simulated stream now and a real
Kafka consumer later (swap the source, keep the scorer).

Run:  python src/serving/stream_consumer.py            # score 500 events
      python src/serving/stream_consumer.py --limit 50 --delay 0.02
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

try:
    from src.serving.inference import InferencePipeline
    from src.serving.stream_producer import stream_bookings
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.serving.inference import InferencePipeline
    from src.serving.stream_producer import stream_bookings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("consumer")

DEFAULT_THRESHOLD = 0.30
PREDICTIONS_LOG = "artifacts/stream_predictions.jsonl"


class StreamingScorer:
    """Scores a stream of booking events and tracks running metrics."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, window: int = 200) -> None:
        self.pipeline = InferencePipeline()
        self.threshold = threshold
        self.n = 0
        self.n_correct = 0
        self.n_labeled = 0
        self.risk_counts = {"low": 0, "medium": 0, "high": 0}
        self.latencies_ms: deque[float] = deque(maxlen=1000)
        # Sliding window of recent (proba, label) — what the drift monitor reads.
        self.recent: deque[tuple[float, int | None]] = deque(maxlen=window)

    def _risk_band(self, p: float) -> str:
        if p < 0.25:
            return "low"
        if p < 0.6:
            return "medium"
        return "high"

    def process_event(self, event: dict) -> dict:
        """Score one event, update running state, return the scored record."""
        t0 = time.perf_counter()
        proba = self.pipeline.predict_proba(event["booking"])
        latency_ms = (time.perf_counter() - t0) * 1000

        pred = int(proba >= self.threshold)
        band = self._risk_band(proba)
        label = event.get("_true_label")

        self.n += 1
        self.latencies_ms.append(latency_ms)
        self.risk_counts[band] += 1
        self.recent.append((proba, label))
        if label is not None:
            self.n_labeled += 1
            self.n_correct += int(pred == label)

        return {
            "event_id": event["event_id"],
            "probability": round(proba, 4),
            "prediction": pred,
            "risk_band": band,
            "true_label": label,
            "latency_ms": round(latency_ms, 2),
        }

    def running_metrics(self) -> dict:
        acc = self.n_correct / self.n_labeled if self.n_labeled else None
        lat = list(self.latencies_ms)
        p50 = sorted(lat)[len(lat) // 2] if lat else 0.0
        p95 = sorted(lat)[int(len(lat) * 0.95)] if lat else 0.0
        return {
            "processed": self.n,
            "running_accuracy": round(acc, 4) if acc is not None else None,
            "risk_distribution": dict(self.risk_counts),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
        }

    def run(self, events: Iterator[dict], report_every: int = 100) -> dict:
        """Consume the whole stream, scoring each event, logging periodic metrics.
        A malformed event is logged and skipped, never fatal."""
        Path("artifacts").mkdir(exist_ok=True)
        with open(PREDICTIONS_LOG, "w", encoding="utf-8") as fh:
            for event in events:
                try:
                    record = self.process_event(event)
                except Exception as exc:
                    log.warning("Skipping event %s: %s", event.get("event_id"), exc)
                    continue
                fh.write(_json_line(record))
                if self.n % report_every == 0:
                    m = self.running_metrics()
                    log.info(
                        "processed=%d | acc=%s | risk=%s | p95=%.2fms",
                        m["processed"],
                        m["running_accuracy"],
                        m["risk_distribution"],
                        m["latency_p95_ms"],
                    )
        final = self.running_metrics()
        log.info("=" * 60)
        log.info("STREAM COMPLETE: %s", final)
        log.info("Per-event predictions written to %s", PREDICTIONS_LOG)
        return final


def _json_line(record: dict) -> str:
    import json

    return json.dumps(record) + "\n"


def main(limit: int, delay: float, threshold: float) -> None:
    scorer = StreamingScorer(threshold=threshold)
    events = stream_bookings(limit=limit, delay=delay, shuffle=True)
    scorer.run(events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="events to score")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between events")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    main(args.limit, args.delay, args.threshold)
