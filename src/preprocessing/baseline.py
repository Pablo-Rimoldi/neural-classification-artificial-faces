"""Baseline correction for EEG data.

Ported from notebook cell 11.
"""
import numpy as np
import pandas as pd


def apply_baseline_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline correction by subtracting pre-stimulus mean.

    For each numeric channel (except Time_ms and Trigger), subtract the mean
    value of that channel during the pre-stimulus period (Time_ms < 0).

    Args:
        df: Input dataframe from load_raw_files with EEG channels and Time_ms.

    Returns:
        A new dataframe with baseline correction applied. Input is not mutated.
    """
    result = df.copy()

    # Get all numeric columns
    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()

    # Exclude Time_ms and Trigger from baseline correction
    exclude_from_baseline = ['Time_ms', 'Trigger']
    channels_to_correct = [c for c in numeric_cols if c not in exclude_from_baseline]

    # Calculate baseline mean from pre-stimulus period (Time_ms < 0)
    baseline_mean = result[result['Time_ms'] < 0][channels_to_correct].mean()

    # Subtract baseline from all samples
    result[channels_to_correct] = result[channels_to_correct] - baseline_mean

    return result
