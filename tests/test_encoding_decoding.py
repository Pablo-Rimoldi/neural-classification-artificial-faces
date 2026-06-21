import numpy as np


def test_encoding_decoding_outputs(tensor):
    from src.analysis.encoding_decoding import run_encoding_decoding
    X, y, subj = tensor
    out = run_encoding_decoding(X, y, subj, make_plot=False)
    n_good = out['n_good']
    assert out['f_map'].shape[0] == n_good
    assert out['K'].shape == (len(out['unique_labels']), n_good, out['f_map'].shape[1])
    assert out['auc_map'].shape == out['K'].shape
    assert 0.0 <= out['decoding_acc_mean'] <= 1.0
    # the constant sex channel (index 18) is flat -> excluded by VAR_THRESHOLD
    assert 18 not in out['good_ch'].tolist()
    assert set(out['unique_labels'].tolist()) == {'AI', 'Real'}
