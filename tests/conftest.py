import numpy as np
import pytest

from src import config


@pytest.fixture(scope='session')
def repo_root():
    return config.PROJECT_ROOT


@pytest.fixture(scope='session')
def tensor():
    # data/file_tensor/ was intentionally removed; build from raw (data/file_raw/).
    if config.TENSOR_PATH.exists():
        z = np.load(config.TENSOR_PATH, allow_pickle=True)
        return z['x'], z['y'], z['subjects']
    # Build via the preprocessing chain (available from Task 7 onward).
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    from src.preprocessing.tensor import build_tensor, save_tensor
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    X, y, subjects = build_tensor(*build_epochs(df))
    save_tensor(X, y, subjects)          # cache for later test sessions
    return X, y, subjects
