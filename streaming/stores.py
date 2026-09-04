"""Where scored transactions get written.

Two implementations behind one interface. Postgres is what runs in the stack;
SQLite is what the tests use, so the pipeline can be exercised end to end
without Docker. Both accept the same rows and answer the same queries.
"""

import sqlite3
from typing import Protocol

from .pipeline import ScoredTransaction

COLUMNS = (
    "trans_id", "cc_num_suffix", "ts", "amt", "category", "merchant_state",
    "risk_score", "flagged", "model_version", "scored_at", "latency_ms", "is_fraud",
)


class ResultStore(Protocol):
    def write(self, scored: ScoredTransaction) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class PostgresStore:
    """Batched inserts into Postgres.

    Writing one row per message would make a network round-trip per transaction
    and cap throughput at a few hundred a second. Rows are buffered and flushed
    together, which is the single biggest throughput lever in this pipeline.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS scored_transactions (
        trans_id       TEXT PRIMARY KEY,
        cc_num_suffix  TEXT        NOT NULL,
        ts             TIMESTAMP   NOT NULL,
        amt            NUMERIC     NOT NULL,
        category       TEXT        NOT NULL,
        merchant_state TEXT        NOT NULL,
        risk_score     DOUBLE PRECISION NOT NULL,
        flagged        BOOLEAN     NOT NULL,
        model_version  TEXT        NOT NULL,
        scored_at      TIMESTAMPTZ NOT NULL,
        latency_ms     DOUBLE PRECISION NOT NULL,
        is_fraud       SMALLINT
    );
    -- The dashboard's two hot queries: newest first, and flagged only.
    CREATE INDEX IF NOT EXISTS idx_scored_at ON scored_transactions (scored_at DESC);
    CREATE INDEX IF NOT EXISTS idx_flagged   ON scored_transactions (flagged) WHERE flagged;
    """

    def __init__(self, dsn: str, batch_size: int = 200):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=False)
        self.batch_size = batch_size
        self._buffer: list[tuple] = []
        with self.conn.cursor() as cur:
            cur.execute(self.SCHEMA)
        self.conn.commit()

    def write(self, scored: ScoredTransaction) -> None:
        row = scored.as_row()
        self._buffer.append(tuple(row[c] for c in COLUMNS))
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        placeholders = ", ".join(["%s"] * len(COLUMNS))
        sql = (
            f"INSERT INTO scored_transactions ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders}) ON CONFLICT (trans_id) DO NOTHING"
        )
        with self.conn.cursor() as cur:
            cur.executemany(sql, self._buffer)
        self.conn.commit()
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()


class SqliteStore:
    """Same interface, no server. Used by the tests and for local runs."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS scored_transactions (
        trans_id       TEXT PRIMARY KEY,
        cc_num_suffix  TEXT    NOT NULL,
        ts             TEXT    NOT NULL,
        amt            REAL    NOT NULL,
        category       TEXT    NOT NULL,
        merchant_state TEXT    NOT NULL,
        risk_score     REAL    NOT NULL,
        flagged        INTEGER NOT NULL,
        model_version  TEXT    NOT NULL,
        scored_at      TEXT    NOT NULL,
        latency_ms     REAL    NOT NULL,
        is_fraud       INTEGER
    );
    """

    def __init__(self, path: str = ":memory:", batch_size: int = 200):
        self.conn = sqlite3.connect(path)
        self.batch_size = batch_size
        self._buffer: list[tuple] = []
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def write(self, scored: ScoredTransaction) -> None:
        row = scored.as_row()
        self._buffer.append(
            tuple(
                # SQLite has no datetime or bool type
                v.isoformat() if hasattr(v, "isoformat")
                else int(v) if isinstance(v, bool)
                else v
                for v in (row[c] for c in COLUMNS)
            )
        )
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        placeholders = ", ".join(["?"] * len(COLUMNS))
        self.conn.executemany(
            f"INSERT OR IGNORE INTO scored_transactions ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})",
            self._buffer,
        )
        self.conn.commit()
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()
