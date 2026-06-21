"""Nested CV, stage-2 focused search, permutation test, and Wilcoxon comparison.

Ported verbatim from notebook cells 56, 57, 59, 60, 62, 63 ("Evaluation" /
"Stage 2" / "Statistical significance testing"). The notebook used
module-level globals `X_flat`/`y_flat`/`subjects_flat`/`trial_ids` and a
module-level `QUICK_TEST` flag; both are replaced here by an explicit `data`
dict argument (as returned by `prepare_ml_data`) and a `quick_test` function
parameter, so the module holds no shared mutable state.
"""
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV,
)
from scipy.stats import wilcoxon

from src import config
from src.models.ml.transforms import (
    aggregate_trials, groups_disjoint, permute_trial_labels,
    summarize_selected_params, representative_params, permutation_p_value,
)
from src.models.ml.models import safe_mcc, acc_scorer, get_focused_grid


def get_scores(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, 'decision_function'):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return model.predict(X).astype(float)


def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5


def run_nested_cv(
    models,
    data,
    *,
    quick_test=True,
    n_outer_folds=5,
    n_outer_repeats=5,
    n_inner_splits=5,
    inner_test_size=0.20,
    collect_params=False,
    verbose=True,
):
    """Repeated group-aware nested CV over `models`.

    `data` is the dict returned by `prepare_ml_data` (keys X_flat, y_flat,
    subjects_flat, trial_ids). When `quick_test`, folds/repeats/splits are
    clamped to (2, 1, 3) and each model's `n_iter` to <=3, exactly as the
    notebook's `QUICK_TEST` global did.

    Returns (results, pooled) or, if `collect_params`, (results, pooled, chosen).
    """
    X_flat = data['X_flat']
    y_flat = data['y_flat']
    subjects_flat = data['subjects_flat']
    trial_ids = data['trial_ids']

    if quick_test:
        n_outer_folds, n_outer_repeats, n_inner_splits = 2, 1, 3

    inner_cv = GroupShuffleSplit(
        n_splits=n_inner_splits, test_size=inner_test_size,
        random_state=config.RANDOM_STATE,
    )

    results, pooled, chosen = {}, {}, {}
    for name, pipeline, param_dist, n_iter in models:
        if quick_test:
            n_iter = min(n_iter, 3)
        t0 = time.time()
        fm = {k: [] for k in ['acc', 'bacc', 'f1', 'mcc', 'auc']}
        p_true, p_pred, p_score = [], [], []
        params_per_fold = []

        for rep in range(n_outer_repeats):
            outer_cv = StratifiedGroupKFold(
                n_splits=n_outer_folds, shuffle=True,
                random_state=config.RANDOM_STATE + rep,
            )
            for tr, te in outer_cv.split(X_flat, y_flat, groups=subjects_flat):
                assert groups_disjoint(subjects_flat[tr], subjects_flat[te])
                search = RandomizedSearchCV(
                    pipeline, param_dist, n_iter=n_iter,
                    scoring=acc_scorer, cv=inner_cv, n_jobs=-1,
                    random_state=config.RANDOM_STATE, error_score=np.nan,
                )
                search.fit(X_flat[tr], y_flat[tr], groups=subjects_flat[tr])
                best = search.best_estimator_
                if collect_params:
                    params_per_fold.append(search.best_params_)
                yt, yp, ys = aggregate_trials(
                    best.predict(X_flat[te]),
                    get_scores(best, X_flat[te]),
                    y_flat[te], trial_ids[te],
                )
                fm['acc'].append(accuracy_score(yt, yp))
                fm['bacc'].append(balanced_accuracy_score(yt, yp))
                fm['f1'].append(f1_score(yt, yp, average='macro'))
                fm['mcc'].append(safe_mcc(yt, yp))
                fm['auc'].append(safe_auc(yt, ys))
                p_true += yt.tolist(); p_pred += yp.tolist(); p_score += ys.tolist()

        results[name] = {k: np.array(v) for k, v in fm.items()}
        pooled[name] = (np.array(p_true), np.array(p_pred), np.array(p_score))
        chosen[name] = params_per_fold
        if verbose:
            print(f"  {name:14s} ACC {np.nanmean(results[name]['acc']):.3f}"
                  f" (var {np.nanvar(results[name]['acc']):.4f})"
                  f"  F1 {np.nanmean(results[name]['f1']):.3f}"
                  f"  AUC {np.nanmean(results[name]['auc']):.3f}"
                  f"  [{time.time() - t0:.0f}s]")

    if collect_params:
        return results, pooled, chosen
    return results, pooled


def select_best(results):
    """Return the name of the model with the highest mean accuracy."""
    order = sorted(results, key=lambda n: -np.nanmean(results[n]['acc']))
    return order[0]


def run_stage2(best_name, models, data, **kw):
    """Focused hyperparameter search on the Stage-1 winner.

    `models` is the Stage-1 list of (name, pipeline, param_dist, n_iter)
    tuples (as from `get_models_and_grids`). Runs nested CV with
    `get_focused_grid(best_name)`; adopts the focused config only if it does
    not worsen mean accuracy. Returns a dict with the adopted/initial grid,
    chosen params per fold, and the comparison results.
    """
    quick_test = kw.get('quick_test', True)
    # `stage1_results` is a control kwarg consumed by this function; it must
    # be popped before forwarding `**kw` to run_nested_cv, which has no such
    # parameter and would raise TypeError otherwise.
    stage1_results = kw.pop('stage1_results', None)
    models_by_name = {m[0]: m for m in models}

    focused_dist, focused_niter = get_focused_grid(best_name)
    winner_pipeline = models_by_name[best_name][1]
    focused_models = [(best_name, winner_pipeline, focused_dist, focused_niter)]

    res_focus, pool_focus, chosen_focus = run_nested_cv(
        focused_models, data, collect_params=True, **kw
    )

    # stage1 baseline must be supplied via kw for fair comparison; if absent,
    # run stage 1 for the winner with the same fold settings.
    if stage1_results is None:
        stage1_kw = {k: v for k, v in kw.items() if k != 'collect_params'}
        res1, _ = run_nested_cv([models_by_name[best_name]], data, **stage1_kw)
        stage1_results = res1

    acc1 = np.nanmean(stage1_results[best_name]['acc'])
    acc2 = np.nanmean(res_focus[best_name]['acc'])

    adopted = bool(acc2 >= acc1)
    if adopted:
        final_grid, final_niter = focused_dist, focused_niter
    else:
        final_grid, final_niter = (
            models_by_name[best_name][2], models_by_name[best_name][3]
        )

    summary = summarize_selected_params(chosen_focus[best_name])

    return {
        'best_name': best_name,
        'adopted': adopted,
        'acc_stage1': acc1,
        'acc_stage2': acc2,
        'final_pipeline': winner_pipeline,
        'final_grid': final_grid,
        'final_niter': final_niter,
        'chosen_params': chosen_focus[best_name],
        'param_summary': summary,
        'results_stage2': res_focus[best_name],
        'pooled_stage2': pool_focus[best_name],
    }


def plot_ml_performance(results, pooled, order, best_name, save_path=None):
    """Boxplot of per-fold accuracy, pooled ROC curves, and confusion matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    box_df = pd.DataFrame({n: results[n]['acc'] for n in order})
    sns.boxplot(data=box_df, ax=axes[0], palette='Set2')
    axes[0].axhline(0.5, color='red', ls='--', label='Chance (accuracy=0.50)')
    axes[0].set_title('Accuracy distribution across folds')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()

    for n in order:
        yt, yp, ys = pooled[n]
        if len(np.unique(yt)) > 1:
            fpr, tpr, _ = roc_curve(yt, ys)
            axes[1].plot(fpr, tpr, label=f"{n} (AUC={auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], 'navy', ls='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Pooled ROC curves')
    axes[1].legend(loc='lower right', fontsize=8)

    yt, yp, _ = pooled[best_name]
    ConfusionMatrixDisplay(
        confusion_matrix(yt, yp), display_labels=['AI', 'Real']
    ).plot(cmap='Blues', ax=axes[2], colorbar=False)
    axes[2].set_title(f'Confusion matrix - {best_name}')

    plt.tight_layout()
    if save_path is None:
        save_path = config.results_path('spatial_pipeline_performance.png')
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def permutation_test(
    best_pipe, data, n_permutations=1000, n_outer_folds=5,
    random_state=config.RANDOM_STATE,
):
    """Permutation test of the final tuned pipeline vs chance.

    Trial-level predictions are pooled over an outer `StratifiedGroupKFold`;
    observed accuracy/MCC are compared against a null distribution built by
    permuting labels at the trial level. p follows Phipson & Smyth (2010).
    """
    X_flat = data['X_flat']
    y_flat = data['y_flat']
    subjects_flat = data['subjects_flat']
    trial_ids = data['trial_ids']

    perm_cv = StratifiedGroupKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=random_state,
    )

    def pooled_trial_preds(estimator, y_target):
        pt, pp = [], []
        for tr, te in perm_cv.split(X_flat, y_target, groups=subjects_flat):
            assert groups_disjoint(subjects_flat[tr], subjects_flat[te])
            est = clone(estimator).fit(X_flat[tr], y_target[tr])
            yt, yp, _ = aggregate_trials(
                est.predict(X_flat[te]), np.zeros(len(te)),
                y_target[te], trial_ids[te],
            )
            pt += yt.tolist(); pp += yp.tolist()
        return np.array(pt), np.array(pp)

    obs_t, obs_p = pooled_trial_preds(best_pipe, y_flat)
    observed_acc = accuracy_score(obs_t, obs_p)
    observed_mcc = safe_mcc(obs_t, obs_p)

    rng = np.random.default_rng(random_state)
    null_mcc, null_acc = [], []
    for _ in range(n_permutations):
        nt, npd = pooled_trial_preds(
            best_pipe, permute_trial_labels(y_flat, trial_ids, rng)
        )
        null_mcc.append(safe_mcc(nt, npd))
        null_acc.append(accuracy_score(nt, npd))
    null_mcc, null_acc = np.array(null_mcc), np.array(null_acc)

    p_acc = permutation_p_value(observed_acc, null_acc)
    p_mcc = permutation_p_value(observed_mcc, null_mcc)

    return {
        'observed_acc': observed_acc,
        'observed_mcc': observed_mcc,
        'p_acc': p_acc,
        'p_mcc': p_mcc,
        'null_acc': null_acc,
        'null_mcc': null_mcc,
    }


def wilcoxon_vs_best(results, best_name):
    """Two-sided Wilcoxon signed-rank test of per-fold accuracy, vs `best_name`,
    Bonferroni-corrected across the other models. Returns a DataFrame indexed
    by the compared model name, with columns p_raw, p_bonferroni, significant.
    """
    order = sorted(results, key=lambda n: -np.nanmean(results[n]['acc']))
    best_folds = results[best_name]['acc']
    others = [n for n in order if n != best_name]
    pvals = []
    for n in others:
        a, b = best_folds, results[n]['acc']
        mask = np.isfinite(a) & np.isfinite(b)
        if np.allclose(a[mask], b[mask]):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a[mask], b[mask], zero_method='zsplit',
                                 alternative='two-sided')
            except ValueError:
                p = 1.0
        pvals.append(p)

    m = len(pvals)
    rows = []
    for n, p in zip(others, pvals):
        pc = min(1.0, p * m)
        rows.append({'model': n, 'p_raw': p, 'p_bonferroni': pc, 'significant': pc < 0.05})
    return pd.DataFrame(rows).set_index('model')
