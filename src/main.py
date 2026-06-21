"""Unified CLI entry point for the EEG classification pipeline.

Composes the stage modules built in Tasks 2-16 (preprocessing -> tensor,
encoding/decoding, ML, DL + XAI) into a single ``run_pipeline`` function and
an ``argparse``-based CLI (``main``), mirroring the notebook's end-to-end
flow (``QUICK_TEST`` / ``RUN_OPTUNA_SEARCH`` globals become explicit
parameters/flags).

Fast mode (``full=False``, ``run_optuna=False``) is the default, matching
the notebook's default ``QUICK_TEST=True`` / ``RUN_OPTUNA_SEARCH=False``
configuration.
"""
import argparse

import numpy as np
import sklearn.base
import torch

from src import config


def _banner(title: str) -> None:
    print(f"\n== {title} ==")


def run_pipeline(
    *,
    full: bool = False,
    run_optuna: bool = False,
    skip_dl: bool = False,
    make_plots: bool = True,
    rebuild_tensor: bool = True,
) -> dict:
    """Run the full notebook flow end to end and return a summary dict.

    Stages, in order:
      1. Preprocess + tensor (Tasks 2-7): rebuild from raw files, or load
         the cached tensor at ``config.TENSOR_PATH``.
      2. Encoding/decoding analysis (Task 8).
      3. ML pipeline: nested CV, stage-2 focused search, permutation test,
         Wilcoxon comparison (Tasks 10-12).
      4. DL pipeline + XAI (Tasks 15-16), unless ``skip_dl``.

    Args:
        full: if True, run full (non quick-test) nested CV / permutation
            counts. If False (default), run in fast/quick-test mode.
        run_optuna: if True, run the Optuna/CMA-ES hyperparameter search for
            the DL model instead of using the precomputed per-fold defaults.
        skip_dl: if True, skip the DL + XAI stage entirely.
        make_plots: if True, save the encoding/decoding, ML performance, and
            XAI figures to ``results/``.
        rebuild_tensor: if True (default), always rebuild the tensor from
            raw files. If False, load the cached tensor at
            ``config.TENSOR_PATH`` when it exists.

    Returns:
        dict summary with at least: 'tensor_shape', 'decoding_acc_mean',
        'best_ml_model', and (when DL is not skipped) DL metrics.
    """
    summary: dict = {}

    # --- 1. Preprocess + tensor ---------------------------------------
    _banner("1. Preprocess + Tensor")
    if rebuild_tensor or not config.TENSOR_PATH.exists():
        from src.io.raw_loader import load_raw_files
        from src.preprocessing.baseline import apply_baseline_correction
        from src.preprocessing.regions import add_region_pca_features, select_columns
        from src.preprocessing.windowing import filter_time_window
        from src.preprocessing.epochs import build_epochs
        from src.preprocessing.tensor import build_tensor, save_tensor

        df = filter_time_window(select_columns(add_region_pca_features(
            apply_baseline_correction(load_raw_files()))))
        X, y, subjects = build_tensor(*build_epochs(df))
        save_tensor(X, y, subjects, config.TENSOR_PATH)
    else:
        from src.preprocessing.tensor import load_tensor
        X, y, subjects = load_tensor(config.TENSOR_PATH)

    summary['tensor_shape'] = tuple(X.shape)

    # --- 2. Encoding / Decoding ----------------------------------------
    _banner("2. Encoding / Decoding")
    from src.analysis.encoding_decoding import run_encoding_decoding

    enc_dec = run_encoding_decoding(
        X, y, subjects,
        make_plot=make_plots,
        save_path=config.results_path('encoding_decoding.png') if make_plots else None,
    )
    summary['decoding_acc_mean'] = enc_dec['decoding_acc_mean']
    summary['decoding_acc_std'] = enc_dec['decoding_acc_std']

    # --- 3. ML pipeline --------------------------------------------------
    _banner("3. Machine Learning")
    from src.models.ml.prepare import prepare_ml_data
    from src.models.ml.models import get_models_and_grids
    from src.models.ml.evaluation import (
        run_nested_cv, select_best, run_stage2, permutation_test,
        wilcoxon_vs_best, plot_ml_performance,
    )
    from src.models.ml.transforms import representative_params

    ml_data = prepare_ml_data(config.TENSOR_PATH)
    models = get_models_and_grids()
    quick_test = not full

    results, pooled = run_nested_cv(models, ml_data, quick_test=quick_test, verbose=True)
    best_name = select_best(results)

    stage2 = run_stage2(best_name, models, ml_data, quick_test=quick_test, verbose=True)

    if make_plots:
        order = sorted(results, key=lambda n: -np.nanmean(results[n]['acc']))
        plot_ml_performance(
            results, pooled, order, best_name,
            save_path=config.results_path('spatial_pipeline_performance.png'),
        )

    final_pipeline = stage2['final_pipeline']
    chosen_params = stage2['chosen_params']
    rep = representative_params(chosen_params)
    best_pipe = sklearn.base.clone(final_pipeline).set_params(**rep)

    n_permutations = 1000 if full else 50
    perm = permutation_test(best_pipe, ml_data, n_permutations=n_permutations)

    wilcoxon_df = wilcoxon_vs_best(results, best_name)

    summary['best_ml_model'] = best_name
    summary['best_ml_acc'] = float(np.nanmean(results[best_name]['acc']))
    summary['ml_stage2_adopted'] = stage2['adopted']
    summary['ml_permutation_p_acc'] = perm['p_acc']
    summary['ml_permutation_p_mcc'] = perm['p_mcc']
    summary['ml_wilcoxon'] = wilcoxon_df

    # --- 4. DL pipeline + XAI -------------------------------------------
    if not skip_dl:
        _banner("4. Deep Learning")
        from src.models.dl.training import load_dl_data, run_dl_nested_cv, summarize_dl_metrics
        from src.models.dl.xai import permutation_importance, gradient_saliency, plot_xai

        dl_data = load_dl_data(config.TENSOR_PATH)
        dl_result = run_dl_nested_cv(dl_data, run_optuna=run_optuna)
        dl_summary = summarize_dl_metrics(dl_result)

        summary['dl_mean_acc'] = dl_summary['mean_acc']
        summary['dl_mean_auc'] = dl_summary['mean_auc']
        summary['dl_p_value'] = dl_summary['p_value']
        summary['dl_significant'] = dl_summary['significant']

        if make_plots:
            _banner("4b. Explainability (XAI)")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            perm_importances, perm_std = permutation_importance(
                dl_result['fold_models'], dl_result['fold_te_indices'],
                dl_data['X_norm'], dl_data['y'], device=device,
            )
            saliency_map, _ = gradient_saliency(
                dl_result['fold_models'], dl_result['fold_te_indices'],
                dl_data['X_norm'], dl_data['y'], device=device,
            )
            plot_xai(
                perm_importances, perm_std, saliency_map,
                n_folds=len(dl_result['fold_models']),
                save_path=config.results_path('xai_analysis.png'),
            )
    else:
        _banner("4. Deep Learning (skipped)")

    return summary


def main() -> None:
    """Argparse CLI wrapping `run_pipeline`. Defaults to fast mode."""
    parser = argparse.ArgumentParser(
        description="Run the EEG artificial-face classification pipeline "
                     "(preprocessing -> tensor -> encoding/decoding -> ML -> DL/XAI).",
    )
    parser.add_argument(
        '--full', action='store_true',
        help="Run the full (non quick-test) nested CV / permutation counts "
             "instead of the fast default.",
    )
    parser.add_argument(
        '--optuna', action='store_true',
        help="Run the Optuna/CMA-ES hyperparameter search for the DL model "
             "instead of using the precomputed per-fold defaults.",
    )
    parser.add_argument(
        '--skip-dl', action='store_true',
        help="Skip the deep-learning + XAI stage entirely.",
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Do not generate or save any figures.",
    )
    parser.add_argument(
        '--use-cached-tensor', action='store_true',
        help="Load the cached tensor at config.TENSOR_PATH instead of "
             "rebuilding it from raw files.",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        full=args.full,
        run_optuna=args.optuna,
        skip_dl=args.skip_dl,
        make_plots=not args.no_plots,
        rebuild_tensor=not args.use_cached_tensor,
    )

    print("\n== Pipeline summary ==")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
