import numpy as np
import pytest

def test_intra_subject_zscore_drops_id_and_normalizes():
    from src.models.ml.transforms import IntraSubjectZScore
    X = np.array([[0., 1., 2.], [0., 3., 6.], [1., 10., 0.], [1., 20., 0.]])
    out = IntraSubjectZScore().fit_transform(X)
    assert out.shape == (4, 2)                        # id column removed
    assert abs(out[:2, 0].mean()) < 1e-9              # per-subject zero mean

def test_decimate_to_odd():
    from src.models.ml.transforms import decimate_to_odd
    X = np.zeros((3, 19, 205))
    assert decimate_to_odd(X, 2).shape[2] % 2 == 1

def test_aggregate_trials_majority_vote():
    from src.models.ml.transforms import aggregate_trials
    yt, yp, ys = aggregate_trials([0,0,1],[0.1,0.2,0.9],[0,0,0],[7,7,7])
    assert yt.tolist()==[0] and yp.tolist()==[0]

def test_permutation_p_value():
    from src.models.ml.transforms import permutation_p_value
    assert permutation_p_value(1.0, [0.0]*9) == (0+1)/(9+1)

def test_models_catalogue():
    from src.models.ml.models import get_models_and_grids, get_focused_grid
    models = get_models_and_grids()
    names = [m[0] for m in models]
    assert names == ['LDA','LinearSVC','LinearSVC_Cal','LogReg_L1','LogReg_EN','SGD','XGBoost','Dummy']
    for _, pipe, grid, n_iter in models:
        assert [s[0] for s in pipe.steps] == ['subjz','scaler','variance','selector','clf']
        assert isinstance(n_iter, int)
    grid, n_iter = get_focused_grid('XGBoost')
    assert isinstance(grid, dict) and n_iter >= 1

def test_prepare_ml_data(tmp_path, tensor):
    import numpy as np
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    d = prepare_ml_data(p, decimation_factor=2)
    assert d['X_flat'].shape[1] == 1 + 18              # id col + 18 channels (sex dropped)
    assert set(np.unique(d['y_ml']).tolist()) <= {0, 1}
    assert '01' not in set(d['subjects_ml'].tolist())  # subject 01 excluded
    assert d['n_timepoints'] % 2 == 1                  # odd after decimate_to_odd


def test_nested_cv_quick_runs(tmp_path, tensor):
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    from src.models.ml.models import get_models_and_grids
    from src.models.ml.evaluation import run_nested_cv, select_best
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    data = prepare_ml_data(p)
    models = [m for m in get_models_and_grids() if m[0] in ('LDA', 'Dummy')]
    results, pooled = run_nested_cv(models, data, quick_test=True, verbose=False)
    assert set(results) == {'LDA', 'Dummy'}
    for name in results:
        assert 'acc' in results[name] and len(results[name]['acc']) >= 1
    assert select_best(results) in {'LDA', 'Dummy'}


def test_run_stage2_both_paths(tmp_path, tensor):
    # Covers run_stage2 with and without an explicit `stage1_results` kwarg.
    # The `stage1_results=...` path is the one documented in the function's
    # own docstring/comment but previously crashed with TypeError because
    # `stage1_results` was forwarded inside **kw to run_nested_cv before
    # being popped.
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    from src.models.ml.models import get_models_and_grids
    from src.models.ml.evaluation import run_nested_cv, run_stage2
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    data = prepare_ml_data(p)
    models = [m for m in get_models_and_grids() if m[0] in ('LDA', 'Dummy')]
    best_name = 'LDA'

    expected_keys = {
        'best_name', 'adopted', 'acc_stage1', 'acc_stage2', 'final_pipeline',
        'final_grid', 'final_niter', 'chosen_params', 'param_summary',
        'results_stage2', 'pooled_stage2',
    }

    # Path 1: no stage1_results supplied -> run_stage2 computes it internally.
    out_no_stage1 = run_stage2(best_name, models, data, quick_test=True, verbose=False)
    assert set(out_no_stage1) == expected_keys
    assert out_no_stage1['best_name'] == best_name

    # Path 2: stage1_results supplied explicitly, as the docstring instructs.
    # This is the path that previously raised TypeError before the fix.
    stage1_results, _ = run_nested_cv(models, data, quick_test=True, verbose=False)
    out_with_stage1 = run_stage2(
        best_name, models, data, quick_test=True, verbose=False,
        stage1_results=stage1_results,
    )
    assert set(out_with_stage1) == expected_keys
    assert out_with_stage1['best_name'] == best_name
    assert out_with_stage1['acc_stage1'] == pytest.approx(
        np.nanmean(stage1_results[best_name]['acc'])
    )


@pytest.mark.slow
def test_nested_cv_full_run_and_permutation(tmp_path, tensor):
    from src.preprocessing.tensor import save_tensor
    from src.models.ml.prepare import prepare_ml_data
    from src.models.ml.models import get_models_and_grids
    from src.models.ml.evaluation import run_nested_cv, select_best, permutation_test
    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'; save_tensor(X, y, subj, p)
    data = prepare_ml_data(p)
    models = get_models_and_grids()
    results, pooled = run_nested_cv(models, data, quick_test=False, verbose=False)
    assert set(results) == {m[0] for m in models}
    best_name = select_best(results)
    assert best_name in results
    best_pipe = [m[1] for m in models if m[0] == best_name][0]
    perm = permutation_test(best_pipe, data, n_permutations=50)
    assert set(perm) == {'observed_acc', 'observed_mcc', 'p_acc', 'p_mcc', 'null_acc', 'null_mcc'}
    assert 0.0 <= perm['p_acc'] <= 1.0
    assert len(perm['null_acc']) == 50
