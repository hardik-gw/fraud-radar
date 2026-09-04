"""Per-card running history, for scoring one transaction at a time.

Training could look backwards through 1.85M rows in memory. A live service
receives one transaction with no context, so it has to *carry* the context.

This module keeps a small summary per card and derives exactly the same history
features that `features.py` computes in batch. The two must agree: if the API
computes `amt_vs_card_mean` even slightly differently from training, the model
receives inputs it has never seen. That failure is called training/serving skew,
it produces no error message, and `tests/test_no_skew.py` exists to catch it.

The rule that keeps them aligned:

    batch:   .shift() before any aggregate, so a row never sees itself
    online:  read features BEFORE update(), so a transaction never sees itself

They are the same rule.
"""

import math
from collections import deque

SEVEN_DAYS = 7 * 24 * 3600
ONE_DAY = 24 * 3600
ONE_HOUR = 3600
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km. Scalar twin of the vectorised version."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class CardHistory:
    """Everything we remember about one card.

    Deliberately small — a bounded deque of recent transactions plus three
    running statistics. Storing every transaction a card ever made would grow
    without limit; this keeps memory flat however long the service runs.
    """

    __slots__ = ("events", "last_lat", "last_long", "last_ts", "m2", "mean", "n")

    def __init__(self):
        self.events: deque = deque()  # (unix_time, amount), last 7 days only
        self.last_ts: float | None = None
        self.last_lat: float | None = None
        self.last_long: float | None = None
        # Welford's online algorithm: running mean and variance without keeping
        # the numbers. Matches pandas' expanding().mean() / .std() exactly.
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    # -- reading -------------------------------------------------------
    def features(
        self, unix_time: float, amount: float, merch_lat: float, merch_long: float
    ) -> dict:
        """History features for a transaction that has NOT been recorded yet."""
        self._evict(unix_time)

        secs_since_prev = (
            unix_time - self.last_ts if self.last_ts is not None else None
        )

        km_from_prev = None
        if self.last_lat is not None:
            km_from_prev = haversine_km(
                self.last_lat, self.last_long, merch_lat, merch_long
            )

        implied_kmh = None
        if km_from_prev is not None and secs_since_prev and secs_since_prev > 0:
            implied_kmh = km_from_prev / (secs_since_prev / 3600.0)

        return {
            "secs_since_prev_txn": secs_since_prev,
            "km_from_prev_txn": km_from_prev,
            "implied_kmh": implied_kmh,
            "txns_prev_1h": self._count_since(unix_time - ONE_HOUR),
            "txns_prev_24h": self._count_since(unix_time - ONE_DAY),
            "txns_prev_7d": self._count_since(unix_time - SEVEN_DAYS),
            "amt_sum_prev_24h": self._sum_since(unix_time - ONE_DAY),
            "amt_vs_card_mean": self.amount_zscore_for(amount),
        }

    def amount_zscore_for(self, amount: float) -> float | None:
        """How unusual `amount` is against this card's prior transactions.

        None until there are two prior transactions, because a standard
        deviation needs at least two points — the same reason pandas'
        expanding std is NaN on a card's first row.
        """
        if self.n < 2:
            return None
        std = math.sqrt(self.m2 / (self.n - 1))
        if std == 0:
            return None
        return (amount - self.mean) / std

    def _count_since(self, cutoff: float) -> int:
        return sum(1 for ts, _ in self.events if ts > cutoff)

    def _sum_since(self, cutoff: float) -> float:
        return sum(amt for ts, amt in self.events if ts > cutoff)

    def _evict(self, now: float) -> None:
        """Drop anything outside the widest window we care about."""
        cutoff = now - SEVEN_DAYS
        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()

    # -- writing -------------------------------------------------------
    def update(self, unix_time: float, amount: float, lat: float, long: float) -> None:
        """Record a transaction. Always called AFTER features() for that same one."""
        self.events.append((unix_time, amount))
        self.last_ts = unix_time
        self.last_lat, self.last_long = lat, long

        # Welford update
        self.n += 1
        delta = amount - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (amount - self.mean)


class CardStateStore:
    """Card number -> CardHistory.

    A plain dict, which is fine for one process and 999 cards. Phase 3 swaps
    this for Redis so several consumers share one view of each card; the two
    methods below are the whole interface that has to be reimplemented.
    """

    def __init__(self):
        self._cards: dict[int, CardHistory] = {}

    def get(self, cc_num: int) -> CardHistory:
        history = self._cards.get(cc_num)
        if history is None:
            history = self._cards[cc_num] = CardHistory()
        return history

    def __len__(self) -> int:
        return len(self._cards)
