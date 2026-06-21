import numpy as np
import pytest
import torch


def test_eeg_dataset_shapes_and_augment_flag():
    from src.models.dl.dataset import EEGDataset
    x = np.random.randn(5, 19, 205).astype('float32')
    y = np.array([0, 1, 0, 1, 0])
    xb, yb = EEGDataset(x, y, augment=False)[0]
    assert tuple(xb.shape) == (19, 205) and yb.dtype == torch.int64
    xb2, _ = EEGDataset(x, y, augment=True)[0]
    assert tuple(xb2.shape) == (19, 205)


def test_stcnn_forward_shape_and_determinism():
    from src.models.dl.architecture import SpatialTemporalCNN
    torch.manual_seed(0)
    m = SpatialTemporalCNN(channels=19, classes=2).eval()
    x = torch.randn(4, 19, 205)
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
    assert tuple(out1.shape) == (4, 2)
    assert torch.allclose(out1, out2)


def test_dl_training_smoke(tmp_path, tensor):
    import torch
    import numpy as np
    from src.preprocessing.tensor import save_tensor
    from src.models.dl.training import load_dl_data, model_factory, train_one_epoch
    from src.models.dl.dataset import EEGDataset
    from torch.utils.data import DataLoader

    X, y, subj = tensor
    p = tmp_path / 'final_tensor.npz'
    save_tensor(X, y, subj, p)
    d = load_dl_data(p)
    assert d['X_norm'].shape[1] == 19 and set(np.unique(d['y']).tolist()) <= {0, 1}

    hp = {'temp_filters': 16, 'kernel_size': 8, 'n_layers': 1, 'adj_init': 'identity',
          'adj_norm': 'sigmoid', 'dropout': 0.3}
    m = model_factory(hp)
    ds = EEGDataset(d['X_norm'][:16], d['y'][:16], augment=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    train_one_epoch(m, dl, opt, torch.nn.CrossEntropyLoss(), torch.device('cpu'))


def test_xai_shapes(tensor):
    import numpy as np
    import torch
    from src.models.dl.architecture import SpatialTemporalCNN
    from src.models.dl.xai import permutation_importance, gradient_saliency, channel_names

    X, y, subj = tensor
    Xn = ((X - X.mean((1, 2), keepdims=True)) / (X.std((1, 2), keepdims=True) + 1e-8)).astype('float32')
    yb = np.isin(y, ['70RM', '80RF']).astype('int64')
    models = [SpatialTemporalCNN(channels=19).eval() for _ in range(2)]
    te = [np.arange(0, 20), np.arange(20, 40)]
    imp, std = permutation_importance(models, te, Xn, yb, torch.device('cpu'), n_repeats=2)
    sal, _ = gradient_saliency(models, te, Xn, yb, torch.device('cpu'))
    assert imp.shape == (19,) and len(channel_names()) == 19
    assert sal.shape[0] == 19 and sal.shape[1] == Xn.shape[2]


@pytest.mark.slow
def test_dl_nested_cv_full_run():
    from src.models.dl.training import load_dl_data, run_dl_nested_cv, summarize_dl_metrics
    from src import config

    data = load_dl_data(config.TENSOR_PATH)
    result = run_dl_nested_cv(data, run_optuna=False, n_folds=5, device=torch.device('cpu'))

    assert len(result['outer_accs']) == 5
    assert len(result['fold_aucs']) == 5
    assert len(result['fold_models']) == 5
    assert len(result['fold_te_indices']) == 5

    mean_acc = float(np.mean(result['outer_accs']))
    assert 0.0 <= mean_acc <= 1.0

    summary = summarize_dl_metrics(result)
    assert 0.0 <= summary['mean_acc'] <= 1.0
    assert 0.0 <= summary['mean_auc'] <= 1.0
    assert 0.0 <= summary['p_value'] <= 1.0
