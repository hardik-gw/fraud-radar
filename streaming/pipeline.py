"""What happens to one transaction between arriving and being stored.

Deliberately knows nothing about Kafka or Postgres. That separation is what
makes the whole pipeline testable without Docker: the same code path runs in
tests/test_pipeline.py against an in-memory queue and SQLite, and in production
against a real broker and a real database.
"""

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ScoredTransaction:
    """One row of the results table."""

    trans_id: str
    cc_num_suffix: str  # last 4 digits only — never store a full card number
    ts: datetime  # when the payment happened
    amt: float
    category: str
    merchant_state: str
    risk_score: float
    flagged: bool
    model_version: str
    scored_at: datetime  # when we scored it
    latency_ms: float
    is_fraud: int | None  # ground truth, available only because we replay a
    # labelled dataset. A real system would not have this at scoring time; it
    # is carried so the Phase 4 dashboard can show live precision and recall.

    def as_row(self) -> dict:
        return asdict(self)


def parse_message(payload: dict) -> dict:
    """Turn a decoded JSON message into what Scorer.score expects.

    JSON has no date type, so timestamps arrive as strings and have to be
    converted back. Getting this wrong is a classic streaming bug: the model
    receives a string where it expects a datetime and fails at feature time,
    one message into the run.
    """
    txn = dict(payload)
    txn["trans_date_trans_time"] = datetime.fromisoformat(
        txn["trans_date_trans_time"]
    )
    # Tolerate a full timestamp as well as a bare date — a producer we don't
    # control may send either, and the model only ever needs the date part.
    txn["dob"] = datetime.fromisoformat(txn["dob"]).date()
    txn["cc_num"] = int(txn["cc_num"])
    return txn


class ScoringPipeline:
    """Scores a message and hands the result to a store."""

    def __init__(self, scorer, store):
        self.scorer = scorer
        self.store = store

    def handle(self, payload: dict) -> ScoredTransaction:
        txn = parse_message(payload)

        started = time.perf_counter()
        result = self.scorer.score(txn)
        latency_ms = (time.perf_counter() - started) * 1000

        scored = ScoredTransaction(
            trans_id=payload["trans_id"],
            cc_num_suffix=str(txn["cc_num"])[-4:],
            ts=txn["trans_date_trans_time"],
            amt=txn["amt"],
            category=txn["category"],
            merchant_state=txn["state"],
            risk_score=result["risk_score"],
            flagged=result["flagged"],
            model_version=result["model_version"],
            scored_at=datetime.now(UTC),
            latency_ms=latency_ms,
            is_fraud=payload.get("is_fraud"),
        )
        self.store.write(scored)
        return scored
