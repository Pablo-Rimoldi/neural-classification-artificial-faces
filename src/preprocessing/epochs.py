"""Temporal cleaning and epoch construction for EEG + PCA tensors.

Ported from notebook cells 22 ("Temporal cleaning") and 24 ("Create lists").

Divergence from the old `tensor_creation.create_lists`: artifact rejection
uses the notebook's EEG-amplitude rule (reject if more than
`config.MAX_BAD_CHANNELS` channels exceed `config.ARTIFACT_THRESHOLD_UV` in
absolute amplitude), NOT the old PCA z-score gate
(`max(abs(pca)) < std(pca) * Z_THRESHOLD`), which has been removed entirely.
"""
import numpy as np
import pandas as pd
from scipy import signal

from src import config


def apply_temporal_cleaning(epoch_data: np.ndarray) -> np.ndarray:
    """Low-pass filter, detrend, and baseline-correct one epoch.

    Args:
        epoch_data: Array of shape (n_times, n_ch).

    Returns:
        Cleaned array of shape (n_times, n_ch): 4th-order Butterworth
        low-pass at 40 Hz (zero-phase via filtfilt, axis=0), linear
        detrend (axis=0), then subtraction of the mean of the first 25
        samples (baseline correction).
    """
    nyq = 0.5 * config.S_FREQ
    b, a = signal.butter(4, 40.0 / nyq, btype='low')
    filtered = signal.filtfilt(b, a, epoch_data, axis=0)
    cleaned = signal.detrend(filtered, axis=0)
    baseline_window = cleaned[:25, :]
    return cleaned - np.mean(baseline_window, axis=0)


def build_epochs(df: pd.DataFrame):
    """Group rows into per-(subject, trigger) epochs, clean, and filter.

    Groups by ['SubjectID', 'TargetCODE'], sorts each group by 'Time_ms',
    applies `apply_temporal_cleaning` to EEG and PCA channels separately,
    truncates/skips to enforce `config.TARGET_EPOCH_SAMPLES`, and rejects
    epochs whose EEG amplitude exceeds `config.ARTIFACT_THRESHOLD_UV` on
    more than `config.MAX_BAD_CHANNELS` channels.

    Args:
        df: Time-filtered dataframe with EEG_CHANNELS, PCA_COLUMNS, and
            metadata columns ('SubjectID', 'TargetCODE', 'SubjectSEX',
            'Time_ms').

    Returns:
        Tuple (EEG_list, PCA_list, y_list, sub_list, subject_sex_list):
        - EEG_list[i]: ndarray (14, TARGET_EPOCH_SAMPLES)
        - PCA_list[i]: ndarray (4, TARGET_EPOCH_SAMPLES)
        - y_list[i]: TargetCODE string
        - sub_list[i]: SubjectID
        - subject_sex_list[i]: 'M'/'F' string
    """
    EEG_list = []
    PCA_list = []
    y_list = []
    sub_list = []
    subject_sex_list = []

    min_samples = max(int(0.05 * config.S_FREQ), 16)

    for (sub_id, trig), group in df.groupby(['SubjectID', 'TargetCODE']):
        group = group.sort_values('Time_ms')
        if len(group) < min_samples:
            continue

        cleaned_eeg = apply_temporal_cleaning(group[config.EEG_CHANNELS].values)
        cleaned_pca = apply_temporal_cleaning(group[config.PCA_COLUMNS].values)

        if cleaned_eeg.shape[0] > config.TARGET_EPOCH_SAMPLES:
            cleaned_eeg = cleaned_eeg[:config.TARGET_EPOCH_SAMPLES, :]
        elif cleaned_eeg.shape[0] < config.TARGET_EPOCH_SAMPLES:
            continue

        if cleaned_pca.shape[0] > config.TARGET_EPOCH_SAMPLES:
            cleaned_pca = cleaned_pca[:config.TARGET_EPOCH_SAMPLES, :]
        elif cleaned_pca.shape[0] < config.TARGET_EPOCH_SAMPLES:
            continue

        channel_max = np.max(np.abs(cleaned_eeg), axis=0)
        if np.sum(channel_max > config.ARTIFACT_THRESHOLD_UV) > config.MAX_BAD_CHANNELS:
            continue

        EEG_list.append(cleaned_eeg.T)
        PCA_list.append(cleaned_pca.T)
        y_list.append(group['TargetCODE'].iloc[0])
        sub_list.append(sub_id)
        subject_sex_list.append(group['SubjectSEX'].iloc[0])

    print(f"Epochs built: {len(EEG_list)}")

    return EEG_list, PCA_list, y_list, sub_list, subject_sex_list
