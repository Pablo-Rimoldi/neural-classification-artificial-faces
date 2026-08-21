import numpy as np
from pathlib import Path
from src import config


def test_constants_match_notebook():
    assert config.S_FREQ == 512
    assert config.TRIGGER_ROW == 75
    assert config.STEP_MS == 1000 / 512
    assert config.EEG_CHANNELS == [
        'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8', 'P3', 'P4',
        'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h',
    ]
    assert config.PCA_COLUMNS == ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']
    assert config.TIME_START_MS == 200 and config.TIME_END_MS == 600
    assert config.TARGET_EPOCH_SAMPLES == 205
    assert config.ARTIFACT_THRESHOLD_UV == 80.0 and config.MAX_BAD_CHANNELS == 2
    assert config.N_SPLITS == 5
    assert config.RIDGE_ALPHAS == [1, 10, 100, 1000, 10000, 100000, 1000000]
    assert config.VAR_THRESHOLD == 1e-6
    assert (config.N250_START, config.N250_END) == (220, 290)
    assert (config.P300_START, config.P300_END) == (280, 500)
    assert config.RANDOM_STATE == 42 and config.DL_SEED == 4204
    assert config.K_CANDIDATES == [5, 10, 15, 'all']
    assert config.CONDITION_TO_BINARY == {'50AM': 0, '60AF': 0, '70RM': 1, '80RF': 1}
    assert config.CONDITION_TO_LABEL == {'50AM': 'AI', '60AF': 'AI', '70RM': 'Real', '80RF': 'Real'}


def test_paths_are_local_not_colab():
    assert 'drive' not in str(config.RAW_DIR).lower()
    assert config.RAW_DIR.name == 'file_raw'
    assert config.PROJECT_ROOT.exists()
    p = config.results_path('test_out.txt')
    assert isinstance(p, Path)
    assert p.parent == config.RESULTS_DIR

