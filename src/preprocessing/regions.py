"""Region-based PCA feature extraction and column selection.

Ported from notebook cell 13. Computes PC1 for each anatomical region
(Frontal, Parietal, Occipital, Temporal) from "remaining" EEG channels
(float64 cols not in EEG_CHANNELS), then selects a fixed set of output columns.

Key note: Region membership is NOT mutually exclusive. A channel matching
multiple prefixes (e.g. 'PO3' matching both P and PO) contributes to both
regions. This is the notebook's behaviour - preserved exactly here.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src import config


def add_region_pca_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PCA features for each anatomical region.

    Computes PC1 for Frontal, Parietal, Occipital, Temporal from remaining
    EEG channels (float64 columns not in EEG_CHANNELS + metadata).

    Region membership is non-exclusive: a channel matching multiple prefixes
    contributes to multiple regions.

    Args:
        df: Baseline-corrected dataframe with EEG channels and metadata.

    Returns:
        DataFrame with original columns + 4 new PCA_* columns.
        Input is not mutated.
    """
    result = df.copy()

    # Define regions by electrode prefix
    regions = {
        'Frontal':   ['F', 'AF', 'FC'],
        'Parietal':  ['P', 'CP'],
        'Occipital': ['PO', 'O'],
        'Temporal':  ['T', 'TP'],
    }

    # Metadata columns (don't participate in PCA)
    base_cols = config.EEG_CHANNELS + ['Trigger', 'SubjectID', 'SubjectSEX',
                                        'TargetCODE', 'TargetNATURE', 'Time_ms']

    # Find "remaining" EEG channels: float64 columns not in base_cols or metadata
    remaining_eeg = [c for c in result.columns
                     if c not in base_cols and result[c].dtype == np.float64]

    # Compute PCA features per region
    pca_features = pd.DataFrame(index=result.index)
    for region, prefixes in regions.items():
        # Non-exclusive membership: channels matching any prefix in this region
        region_cols = [c for c in remaining_eeg if any(c.startswith(p) for p in prefixes)]

        if region_cols:
            # Standardize and apply PCA
            region_data = StandardScaler().fit_transform(result[region_cols])
            pca = PCA(n_components=1)
            pca_features[f'PCA_{region}'] = pca.fit_transform(region_data).flatten()

    # Concatenate PCA features with original dataframe
    result = pd.concat(
        [result.reset_index(drop=True), pca_features.reset_index(drop=True)], axis=1)

    return result


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select a fixed set of output columns.

    Keeps EEG_CHANNELS + PCA_COLUMNS + metadata, but only if present in df.

    Args:
        df: Dataframe with EEG, PCA, and metadata columns.

    Returns:
        DataFrame with only the selected columns (those present in df).
    """
    # Define target columns
    target_cols = (config.EEG_CHANNELS + config.PCA_COLUMNS +
                   ['Trigger', 'SubjectID', 'SubjectSEX', 'TargetCODE', 'TargetNATURE', 'Time_ms'])

    # Keep only columns that exist in df
    cols_to_keep = [c for c in target_cols if c in df.columns]

    return df[cols_to_keep]
