"""Tensor construction, save, and load for the EEG + PCA + sex channel tensor.

Ported verbatim from notebook cell 27 ("Create tensor / save"), with two bug
fixes relative to the old `tensor_creation.create_tensor`/`save_tensor`:

1. Sex channel: the old code filled the sex channel with the raw/encoded
   `SubjectSEX` value (`np.full((1, ...), subject_sex_value)`). The notebook
   instead computes a numeric constant: ``1.0`` if the subject is ``'F'``,
   else ``0.0``.
2. Return typo: the old code returned an undefined name (`subject_final`
   instead of `subjects_final`), which raised a `NameError`.
"""
from pathlib import Path

import numpy as np

from src import config


def build_tensor(eeg_list, pca_list, y_list, sub_list, sex_list=None):
    """Stack EEG, PCA, and a constant FaceSEX channel into a single 3D tensor.

    Args:
        eeg_list: list of ndarrays, each (14, TARGET_EPOCH_SAMPLES).
        pca_list: list of ndarrays, each (4, TARGET_EPOCH_SAMPLES).
        y_list: list of TargetCODE strings ('50AM', '60AF', '70RM', '80RF').
        sub_list: list of SubjectID strings.
        sex_list: optional list of 'M'/'F' SubjectSEX strings (kept for compatibility).

    Returns:
        Tuple (X, y, subjects):
        - X: ndarray (n_epochs, 19, TARGET_EPOCH_SAMPLES), float64. Channel
          order is 14 EEG, 4 PCA, then 1 FaceSEX channel (constant row of 1.0
          if face stimulus is female ('60AF', '80RF') else 0.0).
        - y: ndarray of TargetCODE strings.
        - subjects: ndarray of SubjectID strings.
    """
    combined_X = []
    for j in range(len(eeg_list)):
        # Derive FaceSEX from TargetCODE: '60AF' / '80RF' -> 1.0 (Female), '50AM' / '70RM' -> 0.0 (Male)
        code = str(y_list[j])
        face_sex_val = 1.0 if code.endswith('F') else 0.0
        sex_channel = np.full((1, eeg_list[j].shape[1]), face_sex_val)
        epoch = np.vstack((eeg_list[j], pca_list[j], sex_channel))
        combined_X.append(epoch)

    X_final = np.array(combined_X, dtype=np.float64)
    y_final = np.array(y_list)
    subjects_final = np.array(sub_list)

    _codes, _counts = np.unique(y_final, return_counts=True)
    _label_counts = {str(code): int(count) for code, count in zip(_codes, _counts)}
    print(f"Tensor shape : {tuple(X_final.shape)}  [Epochs x Channels x Time]")
    print(f"Labels       : {_label_counts}")

    return X_final, y_final, subjects_final


def save_tensor(X, y, subjects, path: Path = config.TENSOR_PATH) -> None:
    """Save the tensor, labels, and subject IDs to ``path`` as an .npz file.

    Creates the parent directory of ``path`` if it does not already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=X, y=y, subjects=subjects)
    print(f"Saved to {path}")


def load_tensor(path: Path = config.TENSOR_PATH):
    """Load the tensor, labels, and subject IDs previously saved by `save_tensor`.

    Returns:
        Tuple (X, y, subjects) as stored under the 'x', 'y', 'subjects' keys.
    """
    with np.load(path, allow_pickle=True) as data:
        return data['x'], data['y'], data['subjects']
