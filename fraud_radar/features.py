"""Feature engineering.

The raw columns describe a transaction in isolation. Fraud is rarely visible
that way: a $200 charge is unremarkable on its own and suspicious as the fifth
charge on a card in ten minutes, 400 km from where the card was last used.

So most of the work here builds *history* features — each transaction compared
against what that card did before. Every one of them is strictly backward
looking (`shift()` before any aggregate), because a feature that peeks at the
current or future rows would inflate test scores and then collapse in
production.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

NUMERIC_FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "age_years",
    "city_pop_log",
    "home_to_merchant_km",
    "secs_since_prev_txn",
    "km_from_prev_txn",
    "implied_kmh",
    "txns_prev_1h",
    "txns_prev_24h",
    "txns_prev_7d",
    "amt_sum_prev_24h",
    "amt_vs_card_mean",
]

CATEGORICAL_FEATURES = ["category", "gender", "state"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points, elementwise."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _rolling_prior_counts(df: pd.DataFrame, window: str) -> pd.Series:
    """Per card: how many transactions in the `window` before this one.

    Rolling windows include the current row, so we subtract 1 to leave only the
    prior ones.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, g in df.groupby("cc_num", sort=False):
        counts = pd.Series(1.0, index=g["ts"].to_numpy()).rolling(window).count()
        out.loc[g.index] = counts.to_numpy() - 1.0
    return out


def _rolling_prior_sum(df: pd.DataFrame, window: str) -> pd.Series:
    """Per card: total amount transacted in `window` before this one."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, g in df.groupby("cc_num", sort=False):
        amounts = pd.Series(g["amt"].to_numpy(), index=g["ts"].to_numpy())
        totals = amounts.rolling(window).sum().to_numpy() - g["amt"].to_numpy()
        out.loc[g.index] = totals
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column. Input must be sorted by `ts`."""
    df = df.sort_values(["cc_num", "ts"]).copy()

    # --- the transaction on its own -------------------------------------
    df["log_amt"] = np.log1p(df["amt"])
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    # Card-present fraud skews heavily nocturnal; give the model the shape
    # directly rather than hoping it carves it out of `hour`.
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    df["age_years"] = (df["ts"] - df["dob"]).dt.days / 365.25
    df["city_pop_log"] = np.log1p(df["city_pop"])

    # --- geography -------------------------------------------------------
    # How far the merchant is from the cardholder's home address.
    df["home_to_merchant_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # --- the card's recent history ---------------------------------------
    by_card = df.groupby("cc_num", sort=False)

    df["secs_since_prev_txn"] = by_card["unix_time"].diff()

    prev_lat = by_card["merch_lat"].shift()
    prev_long = by_card["merch_long"].shift()
    df["km_from_prev_txn"] = haversine_km(
        prev_lat, prev_long, df["merch_lat"], df["merch_long"]
    )

    # Implied travel speed between consecutive transactions. A card used in two
    # cities an hour apart implies a speed no traveller achieves — the classic
    # geo-impossibility signal, and something the ULB dataset could not express.
    hours_elapsed = df["secs_since_prev_txn"] / 3600.0
    df["implied_kmh"] = np.where(
        hours_elapsed > 0, df["km_from_prev_txn"] / hours_elapsed, np.nan
    )

    df["txns_prev_1h"] = _rolling_prior_counts(df, "1h")
    df["txns_prev_24h"] = _rolling_prior_counts(df, "24h")
    df["txns_prev_7d"] = _rolling_prior_counts(df, "7d")
    df["amt_sum_prev_24h"] = _rolling_prior_sum(df, "24h")

    # This amount against what the card normally spends. `shift()` first, so the
    # current transaction never contributes to its own baseline.
    prior_mean = by_card["amt"].transform(lambda s: s.shift().expanding().mean())
    prior_std = by_card["amt"].transform(lambda s: s.shift().expanding().std())
    df["amt_vs_card_mean"] = (df["amt"] - prior_mean) / prior_std.replace(0, np.nan)

    return df.sort_values("ts")
