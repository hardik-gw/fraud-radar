# Phase 3 — Streaming Layer · Plan & Status

**Status: code complete, stack unrun.** Everything is written and the pipeline logic is tested end
to end. Kafka and Postgres themselves have never been started, because **Docker is not installed on
this machine**. That is the one outstanding item.

---

## What was built

| File | Role |
|---|---|
| [`streaming/pipeline.py`](../streaming/pipeline.py) | Message → score → row. Knows nothing about Kafka or Postgres. |
| [`streaming/stores.py`](../streaming/stores.py) | `PostgresStore` for the stack, `SqliteStore` for tests. Same interface. |
| [`streaming/producer.py`](../streaming/producer.py) | Replays the 2020 hold-out onto the topic at a set rate, with bursts. |
| [`streaming/consumer.py`](../streaming/consumer.py) | Polls, scores, writes, commits — in that order. |
| [`docker-compose.yml`](../docker-compose.yml) | Kafka (KRaft), Postgres, API, consumer, producer. |
| [`Dockerfile`](../Dockerfile) | One image, three roles. |

The split between `pipeline.py` and the transport is the design decision that mattered. It is why
[`tests/test_pipeline.py`](../tests/test_pipeline.py) can prove the pipeline correct on a laptop with
no Docker: Kafka becomes a list, Postgres becomes SQLite, and every other line is the real one.

---

## Three ideas worth understanding

### 1. Partitioning by card number replaces a whole database

Phase 2 gave each card an in-memory running summary. The obvious objection: run two consumers and
each keeps its own half-blind copy of the same card.

The fix is one argument in the producer — `key=cc_num`. Kafka routes every message with the same key
to the same partition, and each partition is read by exactly one consumer in a group. So a card's
entire history always lands in the same process. **No Redis, no shared cache, no distributed lock.**

This is how real stateful stream processing works, Kafka Streams included. The topic has 4
partitions, which is also the hard ceiling on useful consumers: a fifth would sit idle.

### 2. Commit after the write, never before

```python
store.flush()      # rows are durable
consumer.commit()  # only now do we admit we've read them
```

If the consumer dies between those lines, Kafka redelivers the batch and the rows are written twice.
If they were the other way round, a crash would lose the batch permanently. **Duplicates are
recoverable; gaps are not**, so the order is chosen deliberately.

The duplicates are then absorbed by the primary key on `trans_id` — `INSERT ... ON CONFLICT DO
NOTHING`. At-least-once delivery plus idempotent writes behaves like exactly-once in the table.
There is a test for precisely this.

### 3. Backpressure is not a problem you solve, it's a number you watch

The plan worried about "consumer lag without silently dropping messages". With Kafka there is
nothing to build: the topic is a **log**, not a queue that overflows. A slow consumer just reads
further behind. Lag becomes an observable metric — which is Phase 4's job — rather than lost data.

The producer's random bursts exist to exercise exactly this: a perfectly even stream never lets the
consumer fall behind, so it never proves it can catch up.

---

## What is verified, and what is not

**Verified (15 tests passing):**
- Messages survive JSON serialisation with the right types on the far side
- Streaming scores match the batch pipeline to within 1e-12 across a full card history
- Every fraud on the test card is flagged
- A redelivered batch produces no duplicate rows
- Rows are batched, not written one at a time
- `docker-compose.yml` parses and the dependency graph is right

**Not verified — needs Docker:**
- Kafka actually starting in KRaft mode with this configuration
- The Postgres schema executing against real Postgres (the SQL is Postgres dialect; only the SQLite
  twin has been run)
- The image building, and `uv sync --frozen` succeeding inside it
- Container-to-container networking and the healthcheck gates

Config written but never executed is the least trustworthy kind. Expect one or two things to need
fixing on the first `docker compose up`.

---

## What you have to do

1. **Install Docker Desktop** — https://www.docker.com/products/docker-desktop. Large download,
   needs admin rights.
2. Expect the **Netskope certificate problem** to reappear on image pulls. Docker Desktop has a
   setting for corporate proxies and root certificates under Settings → Resources → Proxies. Same
   root cause as the Kaggle failure: a TLS-inspecting proxy the client doesn't trust yet.
3. Then: `docker compose up --build`, and tell me what breaks.

---

## Reading & watching

### Kafka fundamentals
- 📺 [Apache Kafka in 6 minutes](https://www.youtube.com/watch?v=Ch5VhJzaoaI) — the fastest correct overview
- 📺 [Kafka Topics, Partitions and Offsets](https://www.youtube.com/watch?v=_q1IjK5jjyU)
- 📖 [Kafka docs — Design: the log](https://kafka.apache.org/documentation/#design) — read the section on the log as a storage abstraction; it explains why backpressure isn't a problem here

### Delivery guarantees
- 📖 [Kafka docs — Message Delivery Semantics](https://kafka.apache.org/documentation/#semantics) — at-most-once, at-least-once, exactly-once, and why the third is harder than it sounds
- 📖 [Idempotent consumers](https://microservices.io/patterns/communication-style/idempotent-consumer.html)

### Docker Compose
- 📺 [Docker Compose in 12 minutes](https://www.youtube.com/watch?v=Qw9zlE3t8Ko)
- 📖 [Compose file reference — depends_on and healthchecks](https://docs.docker.com/reference/compose-file/services/#depends_on)
