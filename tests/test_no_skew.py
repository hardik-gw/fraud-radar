"""Prove the online feature code agrees with the batch feature code.

This is the highest-value test in the project. Training/serving skew produces no
exception and no warning — the model simply receives inputs shaped differently
from the ones it learned on, and quietly gets worse. The only way to catch it is
to compute both and compare.

The test replays every real transaction for a handful of cards through
CardHistory, one at a time, exactly as the API would, and checks each history
feature against what features.py computed for that same row in batch.
"""

import math

import pytest

from fraud_radar.data import FEATURE_CACHE, load_features
from fraud_radar.state import CardHistory

pytestmark = pytest.mark.skipif(
    not FEATURE_CACHE.exists(),
    reason="needs the feature cache; run scripts/train.py first",
)

HISTORY_FEATURES = [
    "secs_since_prev_txn",
    "km_from_prev_txn",
    "implied_kmh",
    "txns_prev_1h",
    "txns_prev_24h",
    "txns_prev_7d",
    "amt_sum_prev_24h",
    "amt_vs_card_mean",
]


@pytest.fixture(scope="module")
def sample_cards():
    """Three real cards, every transaction they made, in time order."""
    df = load_features()
    cards = df["cc_num"].drop_duplicates().head(3).tolist()
    return [
        df[df.cc_num == c].sort_values("ts").reset_index(drop=True) for c in cards
    ]


def _matches(batch_value, online_value) -> bool:
    """NaN and None both mean 'no history yet'; otherwise compare numerically."""
    batch_missing = batch_value is None or (
        isinstance(batch_value, float) and math.isnan(batch_value)
    )
    if batch_missing or online_value is None:
        return batch_missing and online_value is None
    return online_value == pytest.approx(float(batch_value), rel=1e-9, abs=1e-9)


def test_online_features_match_batch(sample_cards):
    compared = 0

    for card_rows in sample_cards:
        history = CardHistory()

        for row in card_rows.itertuples():
            unix_time = row.ts.timestamp()

            # Read BEFORE update — the online equivalent of .shift()
            online = history.features(
                unix_time, row.amt, row.merch_lat, row.merch_long
            )

            for name in HISTORY_FEATURES:
                batch_value = getattr(row, name)
                assert _matches(batch_value, online[name]), (
                    f"{name} disagrees on card {row.cc_num} at {row.ts}: "
                    f"batch={batch_value!r} online={online[name]!r}"
                )
                compared += 1

            history.update(unix_time, row.amt, row.merch_lat, row.merch_long)

    assert compared > 10_000, f"only compared {compared} values; sample too small"


def test_history_is_bounded(sample_cards):
    """Memory must not grow with runtime — old events are evicted."""
    card_rows = sample_cards[0]
    history = CardHistory()
    for row in card_rows.itertuples():
        unix_time = row.ts.timestamp()
        history.features(unix_time, row.amt, row.merch_lat, row.merch_long)
        history.update(unix_time, row.amt, row.merch_lat, row.merch_long)

    span_days = (card_rows.ts.max() - card_rows.ts.min()).days
    assert span_days > 300, "expected a card with a long history"
    assert len(history.events) < len(card_rows), "nothing was evicted"
