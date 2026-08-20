import numpy as np
from src import config


def test_temporal_cleaning_shape_and_baseline():
    from src.preprocessing.epochs import apply_temporal_cleaning
    rng = np.random.default_rng(0)
    x = rng.standard_normal((205, 14)) + 5.0
    out = apply_temporal_cleaning(x)
    assert out.shape == (205, 14)
    assert np.abs(out[:25].mean(axis=0)).max() < 1e-6   # first-25 baseline removed


def test_build_epochs_shapes_and_string_labels():
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    eeg, pca, y, sub, sex = build_epochs(df)
    assert len(eeg) == len(pca) == len(y) == len(sub) == len(sex)
    assert eeg[0].shape == (14, config.TARGET_EPOCH_SAMPLES)
    assert pca[0].shape == (4, config.TARGET_EPOCH_SAMPLES)
    assert set(sex) <= {'M', 'F'}                       # strings preserved
    assert all(isinstance(v, str) and len(v) == 4 for v in y)   # TargetCODE strings


def test_build_and_roundtrip_tensor(tmp_path):
    from src.preprocessing.tensor import build_tensor, save_tensor, load_tensor
    eeg = [np.zeros((14, 205)), np.ones((14, 205))]
    pca = [np.zeros((4, 205)),  np.ones((4, 205))]
    X, y, subj = build_tensor(eeg, pca, ['50AM', '60AF'], ['01', '02'], ['F', 'M'])
    assert X.shape == (2, 19, 205) and X.dtype == np.float64
    assert (X[0, 18, :] == 0.0).all()      # '50AM' is Male face -> FaceSEX row = 0.0
    assert (X[1, 18, :] == 1.0).all()      # '60AF' is Female face -> FaceSEX row = 1.0
    p = tmp_path / 'final_tensor.npz'
    save_tensor(X, y, subj, p)
    X2, y2, s2 = load_tensor(p)
    assert np.array_equal(X, X2) and np.array_equal(subj, s2)


def test_full_preprocessing_chain_tensor():
    from src.io.raw_loader import load_raw_files
    from src.preprocessing.baseline import apply_baseline_correction
    from src.preprocessing.regions import add_region_pca_features, select_columns
    from src.preprocessing.windowing import filter_time_window
    from src.preprocessing.epochs import build_epochs
    from src.preprocessing.tensor import build_tensor
    df = filter_time_window(select_columns(add_region_pca_features(
        apply_baseline_correction(load_raw_files()))))
    X, y, subj = build_tensor(*build_epochs(df))
    assert X.ndim == 3 and X.shape[1] == 19 and X.shape[2] == 205
    assert len(set(y.tolist())) == 4
