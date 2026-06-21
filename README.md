# Seeing the Uncanny: Neural Classification of Artificial Faces

## Project Overview

This project investigates the neural representation of AI-generated (GAN), hyper-realistic
faces compared to real human faces. Although the two categories are difficult to tell apart
behaviorally, event-related potentials (ERPs) recorded via EEG can carry a discriminative
signal. This repository implements the full pipeline used to test that hypothesis: from raw
EEG recordings to a tensor representation, through encoding/decoding analysis, classical
Machine Learning, and a Deep Learning model with explainability (XAI).

### Scientific Context

- **P.I.:** Prof. Alice Mado Proverbio (Univ. Milano-Bicocca)
- **Supervision:** Prof. Claudia Casellato (Univ. Pavia)
- **Team:** Pablo Rimoldi, Tommaso Godino, Andrea De Paola, Giacomo Colombo

### Technical Specifications

- **Data source:** 128-channel EEG (10-5 system), 512 Hz sampling rate.
- **Stimuli:** 440 male/female faces (Real vs. GAN-generated).
- **Signal processing:**
  - Bandpass filter: 0.01-70 Hz.
  - Notch filter: 50 Hz.
  - Epochs: -100 ms to 800 ms.
  - Reference: Common Average Reference (CAR).
  - Analysis window: **200-600 ms** post-stimulus.
- **Channels of interest (14):**
  `O1, O2, PO9, PO10, TP7, TP8, P3, P4, AF3, AF4, AFF1h, AFF2h, AFF3h, AFF4h`.

### Classification Task

EEG trials are labeled with one of 4 condition codes, which collapse into a binary
real-vs-AI task for modeling:

| Code | Condition       | Binary class |
|------|------------------|--------------|
| 50   | AI Male (AM)     | AI (0)       |
| 60   | AI Female (AF)   | AI (0)       |
| 70   | Real Male (RM)   | Real (1)     |
| 80   | Real Female (RF) | Real (1)     |

## Repository Layout

The pipeline is implemented as a Python package under `src/`, mirroring the reference
notebook section by section:

```
src/
  config.py                      Global constants and local filesystem paths
  io/
    raw_loader.py                Reads the raw per-subject EEG .txt files into a DataFrame
  preprocessing/
    baseline.py                  Baseline correction of the raw signal
    regions.py                   Region-level PCA features and column selection
    windowing.py                 Restricts samples to the 200-600 ms analysis window
    epochs.py                    Epoch construction and artifact rejection
    tensor.py                    Builds, saves, and loads the final (X, y, subjects) tensor
  analysis/
    encoding_decoding.py         Ridge encoding and decoding analysis on the tensor
  models/
    ml/
      transforms.py              Feature transforms (e.g. parameter representativeness)
      prepare.py                 Loads the tensor into the tabular ML data contract
      models.py                  Model/hyperparameter-grid factory for the ML pipeline
      evaluation.py              Nested CV, stage-2 search, permutation test, plots
    dl/
      dataset.py                 PyTorch Dataset/DataLoader construction
      architecture.py            The STCNN deep learning model definition
      training.py                Nested CV training loop and metric summary for the DL model
      xai.py                     Permutation importance, gradient saliency, XAI plots
  main.py                        Unified CLI entry point composing all stages
```

Legacy, pre-refactor files are kept for reference and are **not** removed; see
[Module map](#module-map) below for how they relate to the new package.

## Pipeline Overview

```
Raw EEG (.txt) -> Preprocessing (baseline, regions, windowing)
                -> Epochs (artifact rejection)
                -> Tensor (X, y, subjects)
                -> Encoding / Decoding analysis
                -> Machine Learning (nested CV over multiple models)
                -> Deep Learning (STCNN, nested CV) + XAI (saliency, permutation importance)
```

Each arrow corresponds to one or more modules under `src/`, composed end to end by
`run_pipeline()` in `src/main.py`.

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Pablo-Rimoldi/neural-classification-artificial-faces
   cd neural-classification-artificial-faces
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**

   ```bash
   python -m src.main
   ```

   By default this runs in **fast mode**: a quick-test configuration with reduced nested-CV
   folds/permutation counts, matching the reference notebook's default
   `QUICK_TEST = True`, `RUN_OPTUNA_SEARCH = False` settings. It rebuilds the tensor from the
   raw files in `data/file_raw/`, runs the encoding/decoding analysis, the ML pipeline, and
   the DL pipeline with XAI, saving figures to `results/`.

### CLI Flags

| Flag                 | Effect                                                                                           |
|-----------------------|---------------------------------------------------------------------------------------------------|
| `--full`             | Run the full (non quick-test) nested CV and permutation-test counts instead of the fast default. |
| `--optuna`           | Run the Optuna/CMA-ES hyperparameter search for the DL model instead of using the precomputed per-fold defaults in `data/best_hyperparameters.json`. |
| `--skip-dl`          | Skip the Deep Learning + XAI stage entirely (preprocessing, encoding/decoding, and ML still run). |
| `--no-plots`         | Do not generate or save any figures.                                                              |
| `--use-cached-tensor`| Load the cached tensor at `data/file_tensor/final_tensor.npz` instead of rebuilding it from raw files. |

Examples:

```bash
# Fast smoke run, ML + DL, no DL hyperparameter search
python -m src.main

# Full run (slow): full nested CV, full permutation counts
python -m src.main --full

# Full run including the Optuna/CMA-ES search for DL hyperparameters
python -m src.main --full --optuna

# Skip Deep Learning entirely (ML-only pipeline)
python -m src.main --skip-dl

# Reuse the tensor already saved on disk instead of rebuilding it from raw EEG files
python -m src.main --use-cached-tensor

# Headless run, no figures written to results/
python -m src.main --no-plots
```

**Fast vs. full mode:** fast mode (default) uses a quick-test configuration intended for
iterating on the code and verifying the pipeline runs end to end in a few minutes. Full mode
(`--full`) reproduces the actual experimental nested cross-validation and permutation-test
counts used for the reported results, and is correspondingly slower (the DL stage in
particular can take a long time on CPU). `--optuna` is independent of `--full` and only
controls whether DL hyperparameters are searched for or loaded from the cached file.

## Running Tests

The test suite mirrors the package layout (one test module per pipeline stage) and marks
long-running tests (full nested CV, Optuna search, DL training) with `@pytest.mark.slow`.

```bash
# Fast suite only (recommended for everyday development)
python -m pytest -m "not slow" -q

# Full suite, including slow tests
python -m pytest -q
```

## Reference Notebook

`notebooks/Neural_classification_artificial_faces.ipynb` is the reference notebook that the
`src/` package mirrors section by section (constants, raw loading, preprocessing, epoching,
tensor construction, encoding/decoding, ML, DL/XAI). It documents the original exploratory
analysis and is kept as the source of truth for the algorithmic steps implemented in `src/`.

## Module Map

The package under `src/` supersedes earlier, exploratory scripts and notebooks. Those files
are kept in the repository for reference and traceability but are no longer the active
implementation:

| Legacy file                                          | Superseded by                                                                                   |
|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `src/preprocessing/preprocessing_eeg.py`              | `src/io/raw_loader.py` + `src/preprocessing/baseline.py`, `regions.py`, `windowing.py`           |
| `src/preprocessing/tensor_creation.py`                | `src/preprocessing/epochs.py` + `src/preprocessing/tensor.py`                                    |
| `src/models/dl/stcnn_nested_cv.py`                    | `src/models/dl/architecture.py`, `dataset.py`, `training.py`                                     |
| `src/preprocessing/data_cleaner.py`                   | Kept as legacy; no direct replacement module (superseded in spirit by the preprocessing package) |
| `src/models/ml/ML_first_draft/ML_01.ipynb`            | Kept as legacy; superseded by `src/models/ml/{transforms,prepare,models,evaluation}.py`          |

These legacy files are intentionally left in place; do not delete them without explicit
confirmation. Placeholder text files (`ruolo*.txt`, `todo`) found alongside some modules are
internal team notes from the refactor and carry no runtime behavior.
