import numpy as np
from src.io.raw_loader import load_raw_files
from src.preprocessing.baseline import apply_baseline_correction
from src import config


def test_baseline_zeroes_prestimulus_mean():
    df = apply_baseline_correction(load_raw_files())
    pre = df[df['Time_ms'] < 0]
    means = pre[config.EEG_CHANNELS].mean().abs()
    assert (means < 1e-6).all()        # pre-stimulus mean ~0 after correction


def test_region_pca_and_selection():
    from src.preprocessing.regions import add_region_pca_features, select_columns
    df = select_columns(add_region_pca_features(apply_baseline_correction(load_raw_files())))
    for c in config.PCA_COLUMNS:
        assert c in df.columns
    assert set(config.EEG_CHANNELS) <= set(df.columns)
    assert df.shape[1] == 14 + 4 + 6        # 14 EEG + 4 PCA + 6 meta(+Trigger)


def test_time_window_bounds():
    from src.preprocessing.windowing import filter_time_window
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'Time_ms': np.array([-50, 0, 200, 400, 600, 700], float)})
    out = filter_time_window(df)
    assert out['Time_ms'].min() >= 200 and out['Time_ms'].max() <= 600
    assert len(out) == 3
