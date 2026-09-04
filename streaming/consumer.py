"""Read transactions off Kafka, score them, and persist the results.

    uv run python -m streaming.consumer

Delivery guarantees, stated plainly because this is where streaming systems
quietly lose data:

* Offsets are committed **after** the database write, never before. If this
  process dies mid-batch, those messages are redelivered on restart rather than
  skipped. That is at-least-once delivery.
* Duplicates from that redelivery are absorbed by the primary key on
  `trans_id` — an insert that has already happened does nothing. At-least-once
  plus idempotent writes is effectively exactly-once in the table.
* Backpressure needs no special handling. Kafka is a log, not a queue that
  overflows: if scoring is slower than production, the consumer simply reads
  further behind, and lag is a number you can watch rather than data you lose.
"""

import json
import logging
import os
import signal
import sys
import time

from fraud_radar.scoring import Scorer

from .pipeline import ScoringPipeline
from .stores import PostgresStore

TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
GROUP_ID = os.getenv("KAFKA_GROUP", "fraud-scorer")
DSN = os.getenv(
    "POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_radar"
)

log = logging.getLogger("consumer")
_running = True


def _stop(signum, frame):
    """Finish the batch in hand, then exit — don't abandon uncommitted work."""
    global _running
    log.info("signal %s received; draining", signum)
    _running = False


def build_consumer():
    from kafka import KafkaConsumer

    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKERS.split(","),
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        enable_auto_commit=False,  # we commit deliberately, after writing
        max_poll_records=200,
        consumer_timeout_ms=1000,
    )


def run() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    scorer = Scorer()
    store = PostgresStore(DSN)
    pipeline = ScoringPipeline(scorer, store)
    consumer = build_consumer()

    log.info("consuming %s as %s, model %s %s",
             TOPIC, GROUP_ID, scorer.model_name, scorer.version)

    processed = flagged = 0
    started = time.monotonic()

    try:
        while _running:
            batches = consumer.poll(timeout_ms=1000, max_records=200)
            if not batches:
                continue

            for records in batches.values():
                for record in records:
                    scored = pipeline.handle(record.value)
                    processed += 1
                    flagged += scored.flagged

            # Write, then commit. Order matters: committing first would mean a
            # crash here loses the batch permanently.
            store.flush()
            consumer.commit()

            if processed % 1000 < 200:
                rate = processed / max(time.monotonic() - started, 1e-9)
                log.info("processed %d (%.0f/s), flagged %d", processed, rate, flagged)
    finally:
        store.close()
        consumer.close()
        log.info("stopped after %d messages, %d flagged", processed, flagged)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s consumer %(message)s", stream=sys.stdout
    )
    run()
