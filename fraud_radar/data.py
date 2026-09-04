"""Loading and splitting the Sparkov transaction data."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Columns we drop immediately. Names, addresses and transaction hashes identify
# people without predicting anything; keeping them invites both leakage and an
# unnecessary privacy footprint.
DROP_COLS = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num",
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
