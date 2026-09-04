"""Turn one incoming transaction into a risk score.

This is the only place that knows how to go from 'a payment happened' to a
number. The API calls it, and in Phase 3 the Kafka consumer will call the same
class directly rather than over HTTP.
"""

import math
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from .features import ALL_FEATURES
from .state import CardStateStore, haversine_km

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "fraud_model_v1.joblib"
)


def to_epoch(ts: datetime) -> float:
    """Seconds since the epoch, treating a naive datetime as UTC.

    `datetime.timestamp()` on a naive value silently uses the *server's* local
    timezone. Two machines in different zones would then compute different gaps
    between the same pair of transactions. Pinning it to UTC makes the score
    depend only on the data.
    """
    return (ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts).timestamp()


class Scorer:
    """Loads the trained pipeline once and scores transactions against it."""

    def __init__(self, model_path: Path | None = None):
        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}. Train one first: uv run python scripts/train.py"
            )
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.threshold: float = artifact["threshold"]
        self.version: str = artifact["version"]
        self.model_name: str = artifact["model_name"]
        self.state = CardStateStore()

    def build_feature_row(self, txn: dict) -> dict:
        """Assemble every feature the model expects for one transaction.

        Order matters: history features are read from the card's state *before*
        this transaction is recorded into it. That mirrors the `.shift()` in
        features.py — a transaction must never contribute to its own baseline.
        """
        ts: datetime = txn["trans_date_trans_time"]
        unix_time = to_epoch(ts)
        history = self.state.get(txn["cc_num"])

        row = {
            # --- the transaction on its own ---
            "amt": txn["amt"],
            "log_amt": math.log1p(txn["amt"]),
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_night": int(ts.hour in (22, 23, 0, 1, 2, 3, 4)),
            "age_years": (ts.date() - txn["dob"]).days / 365.25,
            "city_pop_log": math.log1p(txn["city_pop"]),
            "home_to_merchant_km": haversine_km(
                txn["lat"], txn["long"], txn["merch_lat"], txn["merch_long"]
            ),
            # --- categorical ---
            "category": txn["category"],
            "gender": txn["gender"],
            "state": txn["state"],
        }
        row.update(
            history.features(
                unix_time, txn["amt"], txn["merch_lat"], txn["merch_long"]
            )
        )
        return row

    def score(self, txn: dict, remember: bool = True) -> dict:
        """Score one transaction and, by default, fold it into the card's history."""
        row = self.build_feature_row(txn)
        frame = pd.DataFrame([row], columns=ALL_FEATURES)

        risk_score = float(self.pipeline.predict_proba(frame)[0, 1])

        if remember:
            self.state.get(txn["cc_num"]).update(
                to_epoch(txn["trans_date_trans_time"]),
                txn["amt"],
                txn["merch_lat"],
                txn["merch_long"],
            )

        return {
            "risk_score": risk_score,
            "flagged": risk_score >= self.threshold,
            "threshold": self.threshold,
            "model_version": self.version,
            "model_name": self.model_name,
        }
