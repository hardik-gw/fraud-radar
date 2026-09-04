"""Replay the held-out transactions onto a Kafka topic, as if they were live.

    uv run python -m streaming.producer --rate 50

The dataset is 2020 history. Replaying it in wall-clock time is what makes the
rest of the system a streaming system rather than a batch job.
"""

import argparse
import json
import logging
import os
import random
import time

from fraud_radar.data import load_features

TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# The raw fields a consumer needs; engineered features are the consumer's job,
# exactly as they would be for a real payment event.
#
# The loader renames `trans_date_trans_time` to `ts` for brevity, but the wire
# format keeps the original name so a message is exactly what POST /score
# accepts — one schema, two transports.
WIRE_FIELDS = [
    "cc_num", "amt", "category", "gender", "state",
    "lat", "long", "city_pop", "merch_lat", "merch_long",
]

log = logging.getLogger("producer")


def to_message(row) -> dict:
    msg = {
        "trans_id": row.trans_num,
        "is_fraud": int(row.is_fraud),  # ground truth, for dashboard scoring only
        "trans_date_trans_time": row.ts.isoformat(),
        # date only: dob is a date, and sending it with a midnight time
        # component makes it ambiguous on the far side
        "dob": row.dob.date().isoformat(),
    }
    for field in WIRE_FIELDS:
        value = getattr(row, field)
        msg[field] = (
            value.isoformat() if hasattr(value, "isoformat") else
            int(value) if field in ("cc_num", "city_pop") else
            value if isinstance(value, str) else float(value)
        )
    return msg


def build_producer():
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=BROKERS.split(","),
        value_serializer=lambda v: json.dumps(v).encode(),
        # Keying by card number is the important line in this file. Kafka sends
        # every message with the same key to the same partition, and each
        # partition is read by exactly one consumer. So all of a card's history
        # lands in one consumer's memory, and the per-card state built in
        # Phase 2 stays correct even with several consumers running.
        key_serializer=lambda k: str(k).encode(),
        acks="all",  # don't consider a message sent until replicas have it
        linger_ms=20,  # small batching window; throughput over latency here
    )


def replay(rate: float, limit: int | None, burst_every: int, burst_factor: float,
           dry_run: bool = False):
    df = load_features()
    rows = df[df.split == "test"].sort_values("ts")
    if limit:
        rows = rows.head(limit)

    producer = None if dry_run else build_producer()
    interval = 1.0 / rate
    sent = 0
    started = time.monotonic()
    rng = random.Random(42)
    in_burst_until = 0.0

    for row in rows.itertuples():
        message = to_message(row)
        if producer is not None:
            producer.send(TOPIC, key=message["cc_num"], value=message)
        sent += 1

        # Occasional bursts. A perfectly even stream is a giveaway that nothing
        # real is happening, and it never exercises the consumer's ability to
        # fall behind and catch up — which is the interesting failure mode.
        now = time.monotonic()
        if burst_every and sent % burst_every == 0:
            in_burst_until = now + rng.uniform(1.0, 3.0)
        pace = interval / burst_factor if now < in_burst_until else interval

        if sent % 500 == 0:
            elapsed = now - started
            log.info("sent %d messages (%.0f/s)", sent, sent / max(elapsed, 1e-9))

        time.sleep(pace)

    if producer is not None:
        producer.flush()
        producer.close()
    log.info("done: %d messages in %.1fs", sent, time.monotonic() - started)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=50.0,
                        help="messages per second (default 50)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many messages")
    parser.add_argument("--burst-every", type=int, default=1000,
                        help="start a burst every N messages; 0 disables")
    parser.add_argument("--burst-factor", type=float, default=8.0,
                        help="how many times faster a burst runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="build messages without a broker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s producer %(message)s")
    replay(args.rate, args.limit, args.burst_every, args.burst_factor, args.dry_run)


if __name__ == "__main__":
    main()
