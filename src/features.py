"""
features.py

Turns raw daily weather rows into a supervised learning table:
  X = today's conditions   ->   y = did it rain TOMORROW?

This is the core framing decision of the project: we're not predicting
today's weather (we already know that), we're predicting one day ahead,
which is the genuinely useful, genuinely hard version of the problem.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "temp_f", "humidity_pct", "pressure_hpa", "wind_mph",
    "pressure_drop_1d", "temp_change_1d", "rained_today",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Engineered features: trends often matter more than raw values for
    # weather. A falling pressure is a much stronger rain signal than the
    # absolute pressure number alone.
    df["pressure_drop_1d"] = df["pressure_hpa"].diff()
    df["temp_change_1d"] = df["temp_f"].diff()
    df["rained_today"] = df["rained"]

    # Target: tomorrow's rain outcome, shifted back onto today's row
    df["rained_tomorrow"] = df["rained"].shift(-1)

    df = df.dropna(subset=FEATURE_COLUMNS + ["rained_tomorrow"]).reset_index(drop=True)
    df["rained_tomorrow"] = df["rained_tomorrow"].astype(int)
    return df

def train_test_split_by_time(df: pd.DataFrame, test_fraction: float = 0.2):
    """Time-based split, NOT random shuffling — for time series data,
    shuffling would leak future information into training and make the
    model look better than it really is."""
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
