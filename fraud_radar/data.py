"""Loading and splitting the Sparkov transaction data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_CACHE = PROCESSED_DIR / "features.parquet"

# Columns we drop immediately. Names and addresses identify people without
# predicting anything; keeping them invites both leakage and an unnecessary
# privacy footprint. `trans_num` is kept — it is a random hash, not personal
# data, and the streaming pipeline needs a stable id to make inserts idempotent
# when a message is redelivered.
#
# `unix_time` is dropped for a different reason: it does not agree with
# `trans_date_trans_time`. The offset between them changes on every row, and it
# can advance a full extra day across a month boundary. Everything time-related
# is derived from `ts` instead.
DROP_COLS = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "unix_time",
]


def load_split(split: str) -> pd.DataFrame:
    """Load one of the shipped splits: 'train' (2019-01 to 2020-06) or 'test'."""
    filename = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}[split]
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the dataset first — see the README."
        )

    df = pd.read_csv(path, parse_dates=["trans_date_trans_time", "dob"])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.rename(columns={"trans_date_trans_time": "ts"})
    df["split"] = split
    return df


def load_all() -> pd.DataFrame:
    """Both splits, concatenated and sorted by time.

    Feature engineering runs on the combined frame so that a card's history
    reaches back across the train/test boundary. A test transaction on
    2020-06-22 should be able to see the same card's activity from the previous
    week; splitting first would blank those features out and understate how the
    model performs in production.

    This is not leakage: every engineered feature looks strictly backwards in
    time, and no label from the test period is ever used.
    """
    df = pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    return df.sort_values("ts").reset_index(drop=True)


def load_features(rebuild: bool = False) -> pd.DataFrame:
    """Every transaction with its engineered features, cached.

    Reading and parsing 478 MB of CSV, then rebuilding features, takes the best
    part of a minute on a cold disk. The result is deterministic, so it is
    written to Parquet once and read back in a second or two afterwards.

    Pass rebuild=True after changing anything in features.py, or the cache will
    quietly serve you the old columns.
    """
    from .features import build_features  # imported here to avoid a cycle

    if FEATURE_CACHE.exists() and not rebuild:
        return pd.read_parquet(FEATURE_CACHE)

    df = build_features(load_all())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(FEATURE_CACHE, index=False)
    except Exception as exc:  # noqa: BLE001 - caching is an optimisation
        print(f"warning: could not cache features ({exc}); rebuilding each run")
    return df
