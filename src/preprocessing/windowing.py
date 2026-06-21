"""Time-window filtering for EEG epochs (200-600 ms).

Ported from notebook cell 15.
"""
import pandas as pd
from src import config


def filter_time_window(
    df: pd.DataFrame,
    time_min_ms: float = config.TIME_START_MS,
    time_max_ms: float = config.TIME_END_MS,
) -> pd.DataFrame:
    """Keep rows with time_min_ms <= Time_ms <= time_max_ms.

    Args:
        df: Selected-columns dataframe with 'Time_ms' column.
        time_min_ms: Lower bound (ms), default 200.
        time_max_ms: Upper bound (ms), default 600.

    Returns:
        A copy of df filtered to the time window.
    """
    return df[(df['Time_ms'] >= time_min_ms) & (df['Time_ms'] <= time_max_ms)].copy()
