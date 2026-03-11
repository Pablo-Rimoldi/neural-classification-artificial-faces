import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns

EEG_CHANNELS = [
    'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8',
    'P3', 'P4', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h'
]

SELECTED_COLUMNS = EEG_CHANNELS + [
    'Trigger', 'SubjectID', 'SubjectSEX', 'TargetCODE', 'TargetNATURE', 'SampleIndex'
]

# ─────────────────────────────────────────────
# LOADING DATA
# Prepare new features from each filename: SubjectID, SubjectSEX, TargetCODE,
# TargetNATURE. Also adds a SampleIndex to represent the EEG time-scale.
# ─────────────────────────────────────────────

def load_files(folder_path: str = 'data/Files for ML') -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    print(f"Founded {len(files)} files")

    ls = []
    for file in files:
        f_name = os.path.basename(file).replace('.txt', '')
        temp_df = pd.read_csv(file, sep=r'\s+', engine='python')
        temp_df['SubjectID']    = f_name[:2]
        temp_df['SubjectSEX']   = f_name[2]
        temp_df['TargetCODE']   = f_name[7:]
        temp_df['TargetNATURE'] = f_name[9]
        temp_df['SampleIndex']  = temp_df.index
        ls.append(temp_df)
        print(f"Caricato file: {f_name}")

    dataset = pd.concat(ls, ignore_index=True)
    print(f"Dimensioni totali del dataset: {dataset.shape}")
    return dataset


# ─────────────────────────────────────────────
# COLUMN SELECTION
# Keep only the most meaningful EEG channels plus metadata columns.
# ─────────────────────────────────────────────

def select_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[SELECTED_COLUMNS].copy()


# ─────────────────────────────────────────────
# TIME-WINDOW FILTERING
# Each SampleIndex step = 2 ms (450 steps, window −100 ms → 800 ms).
# Literature suggests focusing on 200 ms – 600 ms → indices 150 – 350.
# ─────────────────────────────────────────────

def filter_time_window(
    dataset: pd.DataFrame,
    sample_min: int = 150,
    sample_max: int = 350,
) -> pd.DataFrame:
    mask = (
        (dataset['SampleIndex'] >= sample_min) &
        (dataset['SampleIndex'] <= sample_max)
    )
    return dataset[mask].copy()


# ─────────────────────────────────────────────
# LABEL ENCODING
# SubjectSEX  : M → 0 | F → 1
# TargetNATURE: R → 0 | A → 1
# ─────────────────────────────────────────────

def encode_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df.loc[df['SubjectSEX'] == 'M', 'SubjectSEX'] = 0
    df.loc[df['SubjectSEX'] == 'F', 'SubjectSEX'] = 1
    df.loc[df['TargetNATURE'] == 'R', 'TargetNATURE'] = 0
    df.loc[df['TargetNATURE'] == 'A', 'TargetNATURE'] = 1
    return df


# ─────────────────────────────────────────────
# EVALUATION WITH LITERATURE
# Plot EEG channels vs. SampleIndex to visually validate the preprocessed data.
# ─────────────────────────────────────────────

def plot_subject(dataset: pd.DataFrame, subject_id: str = '01') -> None:
    report = dataset[dataset['SubjectID'] == subject_id]

    # All channels combined
    plt.figure(figsize=(15, 7))
    for ch in EEG_CHANNELS:
        sns.lineplot(x='SampleIndex', y=ch, data=report, label=ch)
    plt.title(f'EEG Channel Readings — SubjectID: {subject_id}')
    plt.xlabel('Sample Index (Time)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Breakdown by TargetCODE
    for code in report['TargetCODE'].unique():
        subset = report[report['TargetCODE'] == code]
        plt.figure(figsize=(15, 7))
        for ch in EEG_CHANNELS:
            sns.lineplot(x='SampleIndex', y=ch, data=subset, label=ch)
        plt.title(
            f'EEG Channel Readings — SubjectID: {subject_id} | TargetCODE: {code}'
        )
        plt.xlabel('Sample Index (Time)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# Rows per individual sanity-check and save as parquet.
# Note: subject "01" is missing ARTIFICIAL MAN values → 603 rows expected.
# ─────────────────────────────────────────────

def main(output_path: str = 'data/dataset_eeg_preprocessed.parquet') -> pd.DataFrame:
    dataset = load_files()
    dataset = select_columns(dataset)
    dataset = filter_time_window(dataset)
    dataset = encode_labels(dataset)

    plot_subject(dataset, subject_id='01')

    print(dataset['SubjectID'].value_counts())
    dataset.to_parquet(output_path)
    print(f"Dataset saved to {output_path}")
    return dataset


if __name__ == '__main__':
    main()
