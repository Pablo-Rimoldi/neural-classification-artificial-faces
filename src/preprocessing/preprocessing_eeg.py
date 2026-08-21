import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EEG_CHANNELS = [
    'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8',
    'P3', 'P4', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h'
]

METADATA_COLUMNS = ['Trigger', 'SubjectID', 'SubjectSEX', 'TargetCODE', 'TargetNATURE', 'Time_ms']
PCA_COLUMNS = ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']

# ─────────────────────────────────────────────
# LOADING DATA
# Prepare new features from each filename: SubjectID, SubjectSEX, TargetCODE,
# TargetNATURE. Also adds Time_ms using sampling frequency and trigger row.
# ─────────────────────────────────────────────

def load_files(
    folder_path: str = 'data/file_raw',
    sfreq: int = 512,
    trigger_row: int = 75,
) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    print(f"Found {len(files)} files")

    step_ms = 1000 / sfreq

    ls = []
    for file in files:
        f_name = os.path.basename(file).replace('.txt', '')
        temp_df = pd.read_csv(file, sep=r'\s+', engine='python')
        temp_df['SubjectID']    = f_name[:2]
        temp_df['SubjectSEX']   = f_name[2]
        temp_df['TargetCODE']   = f_name[7:]
        temp_df['TargetNATURE'] = f_name[9]
        temp_df['Time_ms'] = (np.arange(len(temp_df)) - (trigger_row - 1)) * step_ms
        ls.append(temp_df)
        print(f"Caricato file: {f_name}")

    dataset = pd.concat(ls, ignore_index=True)
    print(f"Dimensioni totali del dataset: {dataset.shape}")
    return dataset


# ─────────────────────────────────────────────
# BASELINE CORRECTION
# Subtracts pre-trigger mean (Time_ms < 0) from numeric EEG channels.
# ─────────────────────────────────────────────

def apply_baseline_correction(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols_from_baseline = ['Time_ms', 'Trigger']
    channels_to_correct = [col for col in numeric_cols if col not in exclude_cols_from_baseline]

    baseline_mask = df['Time_ms'] < 0
    baseline_mean = df.loc[baseline_mask, channels_to_correct].mean()

    df.loc[:, channels_to_correct] = df[channels_to_correct] - baseline_mean
    print("Baseline mean (first 10 channels):")
    print(baseline_mean.head(10))
    return df


# ─────────────────────────────────────────────
# REGION-BASED PCA
# Groups non-selected channels by EEG region and adds 1 PCA component per region.
# ─────────────────────────────────────────────

def add_region_pca_features(
    dataset: pd.DataFrame,
    reduced_columns: list[str] | None = None,
) -> pd.DataFrame:
    df = dataset.copy()
    reduced_columns = reduced_columns or EEG_CHANNELS
    selected_columns = reduced_columns + METADATA_COLUMNS

    regions = {
        'Frontal': ['F', 'AF', 'FC'],
        'Parietal': ['P', 'CP'],
        'Occipital': ['O', 'PO'],
        'Temporal': ['T', 'TP'],
    }

    remaining_cols = [
        col for col in df.columns
        if col not in selected_columns and np.issubdtype(df[col].dtype, np.floating)
    ]

    pca_features = pd.DataFrame(index=df.index)
    for region, prefixes in regions.items():
        region_cols = [col for col in remaining_cols if any(col.startswith(p) for p in prefixes)]
        if not region_cols:
            continue

        print(f"Processing {region} region with {len(region_cols)} channels")
        region_data = StandardScaler().fit_transform(df[region_cols])
        pca = PCA(n_components=1)
        pca_features[f'PCA_{region}'] = pca.fit_transform(region_data)
        print(f"Explained variance for {region} (PC1): {pca.explained_variance_ratio_[0]:.4f}")

    if not pca_features.empty:
        df = pd.concat([df.reset_index(drop=True), pca_features.reset_index(drop=True)], axis=1)

    return df


# ─────────────────────────────────────────────
# COLUMN SELECTION
# Keep reduced channels + metadata + PCA region features.
# ─────────────────────────────────────────────

def select_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    desired = EEG_CHANNELS + PCA_COLUMNS + METADATA_COLUMNS
    available = [col for col in desired if col in dataset.columns]
    return dataset[available].copy()


# ─────────────────────────────────────────────
# TIME-WINDOW FILTERING
# Literature suggests focusing on 200 ms – 600 ms in post-trigger window.
# ─────────────────────────────────────────────

def filter_time_window(
    dataset: pd.DataFrame,
    time_min_ms: int = 200,
    time_max_ms: int = 600,
) -> pd.DataFrame:
    mask = (
        (dataset['Time_ms'] >= time_min_ms) &
        (dataset['Time_ms'] <= time_max_ms)
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
# Plot EEG channels vs. Time_ms to visually validate the preprocessed data.
# ─────────────────────────────────────────────

def plot_subject(dataset: pd.DataFrame, subject_id: str = '01') -> None:
    report = dataset[dataset['SubjectID'] == subject_id]
    if report.empty:
        print(f"No rows found for SubjectID={subject_id}")
        return

    # All channels combined
    plt.figure(figsize=(15, 7))
    for ch in EEG_CHANNELS:
        sns.lineplot(x='Time_ms', y=ch, data=report, label=ch)
    plt.title(f'EEG Channel Readings — SubjectID: {subject_id}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Breakdown by TargetCODE
    for code in report['TargetCODE'].unique():
        subset = report[report['TargetCODE'] == code]
        plt.figure(figsize=(15, 7))
        for ch in EEG_CHANNELS:
            sns.lineplot(x='Time_ms', y=ch, data=subset, label=ch)
        plt.title(
            f'EEG Channel Readings — SubjectID: {subject_id} | TargetCODE: {code}'
        )
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# Rows per individual sanity-check and save as parquet.
# Note: subject "01" is missing ARTIFICIAL MAN values → 603 rows expected.
# ─────────────────────────────────────────────

def main(
    input_folder: str = 'data/file_raw',
    output_path: str = 'data/dataset_eeg_preprocessed.parquet',
) -> pd.DataFrame:
    dataset = load_files(input_folder)
    dataset = apply_baseline_correction(dataset)
    dataset = add_region_pca_features(dataset)
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
