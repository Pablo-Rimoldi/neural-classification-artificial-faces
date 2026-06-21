"""Global constants and local filesystem paths for the EEG classification pipeline.

Ported verbatim from notebook cell 5 ("## 1. Global Constants"). Colab/Drive
paths have been replaced with local repository paths.
"""
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, RobustScaler

# --- Paths (local) ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'file_raw'
PARQUET_PATH = DATA_DIR / 'dataset_eeg_preprocessed.parquet'
TENSOR_PATH = DATA_DIR / 'file_tensor' / 'final_tensor.npz'
BEST_HP_PATH = DATA_DIR / 'best_hyperparameters.json'
RESULTS_DIR = PROJECT_ROOT / 'results'

# --- Acquisition / signal constants ----------------------------------------
S_FREQ = 512
TRIGGER_ROW = 75
STEP_MS = 1000 / S_FREQ

EEG_CHANNELS = [
    'O1', 'O2', 'PO9', 'PO10', 'TP7', 'TP8', 'P3', 'P4',
    'AF3', 'AF4', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h',
]
PCA_COLUMNS = ['PCA_Frontal', 'PCA_Parietal', 'PCA_Occipital', 'PCA_Temporal']

# --- Epoching / artifact rejection ------------------------------------------
TIME_START_MS = 200
TIME_END_MS = 600
TARGET_EPOCH_SAMPLES = 205
ARTIFACT_THRESHOLD_UV = 80.0
MAX_BAD_CHANNELS = 2

# --- Encoding / decoding analysis -------------------------------------------
N_SPLITS = 5
RIDGE_ALPHAS = [1, 10, 100, 1000, 10000, 100000, 1000000]
VAR_THRESHOLD = 1e-6
N250_START, N250_END = 220, 290
P300_START, P300_END = 280, 500

# --- ML pipeline -------------------------------------------------------------
RANDOM_STATE = 42
K_CANDIDATES = [5, 10, 15, 'all']
SCALER_CHOICES = ['passthrough', MinMaxScaler(), RobustScaler()]

# --- DL pipeline -------------------------------------------------------------
DL_SEED = 3407

# --- Label maps (verbatim) ---------------------------------------------------
# Condition codes -> binary class: {'50AM', '60AF'} -> 0 (AI), {'70RM', '80RF'} -> 1 (Real).
CONDITION_TO_BINARY = {'50AM': 0, '60AF': 0, '70RM': 1, '80RF': 1}
# String labels for encoding/decoding analyses.
CONDITION_TO_LABEL = {'50AM': 'AI', '60AF': 'AI', '70RM': 'Real', '80RF': 'Real'}


def results_path(name: str) -> Path:
    """Return ``RESULTS_DIR / name``, creating ``RESULTS_DIR`` if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
