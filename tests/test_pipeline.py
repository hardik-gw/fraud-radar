"""End-to-end streaming test with the brokers swapped out.

Kafka and Postgres are replaced by a plain list and SQLite. Everything between
— message serialisation, parsing, per-card state, scoring, batched writes,
idempotency — is the same code that runs in the real stack.

This is why `pipeline.py` knows nothing about Kafka: the interesting logic can
be proven correct on a laptop with no Docker installed.
"""

import json

import pytest

from fraud_radar.data import FEATURE_CACHE, load_features
from fraud_radar.features import ALL_FEATURES
from fraud_radar.scoring import DEFAULT_MODEL_PATH, Scorer
from streaming.pipeline import ScoringPipeline
from streaming.producer import to_message
from streaming.stores import SqliteStore

pytestmark = pytest.mark.skipif(
    not (FEATURE_CACHE.exists() and DEFAULT_MODEL_PATH.exists()),
    reason="needs the feature cache and a trained model",
)


@pytest.fixture(scope="module")
def replay_rows():
    """One real card's full history, including the frauds in it."""
    df = load_features()
    fraud_cards = df[(df.split == "test") & (df.is_fraud == 1)].cc_num
    card = fraud_cards.iloc[0]
    return df[df.cc_num == card].sort_values("ts").reset_index(drop=True)


def wire_roundtrip(row) -> dict:
    """Producer -> JSON bytes -> consumer, exactly as Kafka would carry it."""
    return json.loads(json.dumps(to_message(row)).encode().decode())


def test_messages_survive_json_serialisation(replay_rows):
    """Timestamps and card numbers must come back as the right types.

    JSON has no date type. If the consumer forgets to parse them back, feature
    building fails one message into the run — so this is checked first.
    """
    msg = wire_roundtrip(next(replay_rows.itertuples()))
    assert isinstance(msg["cc_num"], int)
    assert isinstance(msg["trans_date_trans_time"], str)
    assert isinstance(msg["amt"], float)
    assert set(msg) >= {"trans_id", "is_fraud", "cc_num", "dob", "merch_lat"}


def test_streamed_scores_match_the_batch_pipeline(replay_rows):
    """The whole point: streaming must not change the answer.

    Every transaction goes through serialisation, parsing and the online state
    store, then the results are compared against scoring the same rows directly
    from the trained pipeline.
    """
    scorer = Scorer()
    batch_scores = scorer.pipeline.predict_proba(replay_rows[ALL_FEATURES])[:, 1]

    store = SqliteStore(batch_size=50)
    pipeline = ScoringPipeline(Scorer(), store)

    streamed = [pipeline.handle(wire_roundtrip(r)).risk_score
                for r in replay_rows.itertuples()]
    store.flush()

    for i, (streamed_score, batch_score) in enumerate(zip(streamed, batch_scores)):
        assert streamed_score == pytest.approx(batch_score, abs=1e-12), (
            f"row {i} disagrees: streamed {streamed_score} vs batch {batch_score}"
        )

    rows = store.conn.execute("SELECT COUNT(*) FROM scored_transactions").fetchone()
    assert rows[0] == len(replay_rows), "not every message reached the table"
    store.close()


def test_frauds_are_flagged(replay_rows):
    store = SqliteStore()
    pipeline = ScoringPipeline(Scorer(), store)
    for r in replay_rows.itertuples():
        pipeline.handle(wire_roundtrip(r))
    store.flush()

    caught, total = store.conn.execute(
        "SELECT SUM(flagged), COUNT(*) FROM scored_transactions WHERE is_fraud = 1"
    ).fetchone()
    assert total > 0, "test card has no fraud to catch"
    assert caught == total, f"only flagged {caught} of {total} frauds"
    store.close()


def test_redelivered_messages_do_not_duplicate_rows(replay_rows):
    """At-least-once delivery means the same message can arrive twice.

    The primary key on trans_id absorbs it. Without this, a consumer restart
    would double-count every flagged transaction on the dashboard.
    """
    store = SqliteStore(batch_size=10)
    pipeline = ScoringPipeline(Scorer(), store)

    messages = [wire_roundtrip(r) for r in replay_rows.head(40).itertuples()]
    for msg in messages:
        pipeline.handle(msg)
    for msg in messages:  # the whole batch redelivered after a crash
        pipeline.handle(msg)
    store.flush()

    count = store.conn.execute(
        "SELECT COUNT(*) FROM scored_transactions"
    ).fetchone()[0]
    assert count == len(messages), f"{count} rows from {len(messages)} unique messages"
    store.close()


def test_store_batches_instead_of_writing_every_row(replay_rows):
    """Rows should sit in the buffer until the batch is full."""
    store = SqliteStore(batch_size=25)
    pipeline = ScoringPipeline(Scorer(), store)

    for r in replay_rows.head(10).itertuples():
        pipeline.handle(wire_roundtrip(r))
    assert store.conn.execute(
        "SELECT COUNT(*) FROM scored_transactions"
    ).fetchone()[0] == 0, "wrote before the batch was full"

    store.flush()
    assert store.conn.execute(
        "SELECT COUNT(*) FROM scored_transactions"
    ).fetchone()[0] == 10
    store.close()
