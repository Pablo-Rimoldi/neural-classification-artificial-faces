import numpy as np
from src import config


def test_constants_match_notebook():
    assert config.S_FREQ == 512
    assert config.EEG_CHANNELS[0] == 'O1' and len(config.EEG_CHANNELS) == 14
    assert config.PCA_COLUMNS == ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']
    assert config.ARTIFACT_THRESHOLD_UV == 80.0 and config.MAX_BAD_CHANNELS == 2
    assert config.RANDOM_STATE == 42 and config.DL_SEED == 3407
    assert config.TARGET_EPOCH_SAMPLES == 205


def test_paths_are_local_not_colab():
    assert 'drive' not in str(config.RAW_DIR).lower()
    assert config.RAW_DIR.name == 'file_raw'
