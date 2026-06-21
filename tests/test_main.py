"""Tests for the unified pipeline entry point (src/main.py)."""
import pytest


def test_public_api_imports():
    """Every public module in the refactored src package must be importable.

    This is the integration gate for the package layout documented in
    README.md (see "Repository layout" / "Module map").
    """
    import importlib

    for mod in [
        'src.config',
        'src.io.raw_loader',
        'src.preprocessing.baseline',
        'src.preprocessing.regions',
        'src.preprocessing.windowing',
        'src.preprocessing.epochs',
        'src.preprocessing.tensor',
        'src.analysis.encoding_decoding',
        'src.models.ml.transforms',
        'src.models.ml.prepare',
        'src.models.ml.models',
        'src.models.ml.evaluation',
        'src.models.dl.dataset',
        'src.models.dl.architecture',
        'src.models.dl.training',
        'src.models.dl.xai',
        'src.main',
    ]:
        importlib.import_module(mod)


def test_run_pipeline_fast_smoke(tmp_path, tensor, monkeypatch):
    import numpy as np
    from src import config
    from src.preprocessing.tensor import save_tensor
    # point config at a temp tensor so we don't rebuild from raw
    p = tmp_path / 'final_tensor.npz'
    save_tensor(*tensor, p)
    monkeypatch.setattr(config, 'TENSOR_PATH', p)
    from src.main import run_pipeline
    summary = run_pipeline(full=False, run_optuna=False, skip_dl=True,
                           make_plots=False, rebuild_tensor=False)
    assert 'decoding_acc_mean' in summary and 'best_ml_model' in summary
    assert summary['tensor_shape'][1] == 19
