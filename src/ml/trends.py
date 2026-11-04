import pandas as pd
import numpy as np


def run_trend_analysis(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Aggregates daily sales, computes 7-day rolling statistics, and flags spikes."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date

    daily = data.groupby("date").agg(
        daily_amount=("amount", "sum"),
        daily_txns=("transaction_id", "count")
    ).reset_index().sort_values("date")

    # Rolling window stats
    daily["rolling_mean"] = daily["daily_amount"].rolling(window=7, min_periods=3).mean()
    daily["rolling_std"] = daily["daily_amount"].rolling(window=7, min_periods=3).std().replace(0, np.nan)

    daily["z_score"] = (daily["daily_amount"] - daily["rolling_mean"]) / daily["rolling_std"]
    daily["z_score"] = daily["z_score"].fillna(0.0)

    daily["is_spike"] = (daily["z_score"] >= z_threshold).astype(int)
    return daily