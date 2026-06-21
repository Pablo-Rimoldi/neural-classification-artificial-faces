"""ML data preparation: tensor -> flattened design matrix with subject codes."""

import numpy as np
from pathlib import Path

from src import config
from src.preprocessing.tensor import load_tensor
from src.models.ml.transforms import decimate_to_odd, spatial_flatten


def prepare_ml_data(
    tensor_path=None,
    decimation_factor=2,
    exclude_subjects=('01', 1)
):
    """
    Load tensor, apply preprocessing, and return ML-ready design matrix.

    Parameters
    ----------
    tensor_path : Path or str, optional
        Path to tensor .npz file. Defaults to config.TENSOR_PATH.
    decimation_factor : int, default=2
        Factor for temporal decimation.
    exclude_subjects : tuple, default=('01', 1)
        Subject identifiers to exclude.

    Returns
    -------
    dict
        Keys:
        - X_flat: (n_rows, 1+n_channels) with col 0 = subject code
        - y_flat: (n_rows,) labels at timepoint resolution
        - subjects_flat: (n_rows,) subject codes at timepoint resolution
        - trial_ids: (n_rows,) trial indices at timepoint resolution
        - y_ml: (n_trials,) aggregated labels (0=AI, 1=Real)
        - subjects_ml: (n_trials,) subject codes at trial resolution
        - n_trials: int
        - n_timepoints: int (odd after decimation)
    """
    if tensor_path is None:
        tensor_path = config.TENSOR_PATH
    tensor_path = Path(tensor_path)

    # Load tensor
    _data = np.load(tensor_path, allow_pickle=True)
    X_ml_raw = _data['x']
    y_ml_codes = _data['y']
    subjects_ml = _data['subjects']

    # Exclude subjects
    EXCLUDE = set(exclude_subjects)
    keep = ~np.isin(subjects_ml, list(EXCLUDE))
    X_ml_raw = X_ml_raw[keep]
    y_ml_codes = y_ml_codes[keep]
    subjects_ml = subjects_ml[keep]

    # Drop sex channel if present
    if X_ml_raw.shape[1] == 19:
        X_ml_raw = X_ml_raw[:, :18, :]

    # Map condition codes to binary labels (0=AI, 1=Real)
    y_ml = np.array([config.CONDITION_TO_BINARY[code] for code in y_ml_codes])

    # Decimate temporal axis
    X_ml_dec = decimate_to_odd(X_ml_raw, decimation_factor)

    # Spatial flatten (trial x tp x channel -> (trial*tp) x channel)
    X_flat, y_flat, subjects_flat, trial_ids = spatial_flatten(
        X_ml_dec, y_ml, subjects_ml)

    # Prepend subject-code column
    _, subj_codes = np.unique(subjects_flat, return_inverse=True)
    X_flat = np.column_stack([subj_codes.astype(float), X_flat])

    return {
        'X_flat': X_flat,
        'y_flat': y_flat,
        'subjects_flat': subjects_flat,
        'trial_ids': trial_ids,
        'y_ml': y_ml,
        'subjects_ml': subjects_ml,
        'n_trials': X_ml_dec.shape[0],
        'n_timepoints': X_ml_dec.shape[2],
    }
