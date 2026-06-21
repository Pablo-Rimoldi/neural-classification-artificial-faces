"""Raw EEG data loader from .txt files with filename feature engineering."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def load_raw_files(
    folder: Path | str = config.RAW_DIR,
    sfreq: int = config.S_FREQ,
    trigger_row: int = config.TRIGGER_ROW,
) -> pd.DataFrame:
    """Load raw EEG .txt files and engineer metadata columns from filenames.

    Each filename follows the pattern: SSXMMTTCCC.txt where:
    - SS: SubjectID (2 chars, e.g. '01')
    - X: SubjectSEX (1 char, 'M' or 'F')
    - MM: reserved (2 chars)
    - TT: reserved (2 chars)
    - C: TargetNATURE (1 char, 'R' or 'A')
    - CC: reserved (2 chars)

    The TargetCODE is extracted as f_name[7:] (e.g. '50AM', '60AF', '70RM', '80RF').

    Time_ms is anchored at the trigger row (trigger_row - 1), with negative values
    for pre-trigger samples and positive for post-trigger.

    Args:
        folder: Path to folder containing raw .txt files. Defaults to config.RAW_DIR.
        sfreq: Sampling frequency in Hz. Defaults to config.S_FREQ (512).
        trigger_row: Row index of trigger event (1-indexed). Defaults to config.TRIGGER_ROW (75).

    Returns:
        DataFrame with all loaded raw data concatenated and metadata columns added:
        - SubjectID (str, 2 chars)
        - SubjectSEX (str, 'M' or 'F')
        - TargetCODE (str, e.g. '50AM')
        - TargetNATURE (str, 'R' or 'A')
        - Time_ms (float, milliseconds relative to trigger)
    """
    folder = Path(folder) if isinstance(folder, str) else folder
    files = sorted(glob.glob(os.path.join(str(folder), '*.txt')))
    print(f"Found {len(files)} files")

    step_ms = 1000 / sfreq

    rows = []
    for file in files:
        f_name = os.path.basename(file).replace('.txt', '')
        temp_df = pd.read_csv(file, sep=r'\s+', engine='python')
        temp_df['SubjectID'] = f_name[:2]
        temp_df['SubjectSEX'] = f_name[2]
        temp_df['TargetCODE'] = f_name[7:]
        temp_df['TargetNATURE'] = f_name[9]
        temp_df['Time_ms'] = (np.arange(len(temp_df)) - (trigger_row - 1)) * step_ms
        rows.append(temp_df)

    dataset_raw = pd.concat(rows, ignore_index=True)
    print(f"Dataset shape: {dataset_raw.shape}")
    return dataset_raw
