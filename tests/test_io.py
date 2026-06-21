from src.io.raw_loader import load_raw_files
from src import config

def test_load_raw_files_shape_and_metadata():
    df = load_raw_files()
    assert {'SubjectID','SubjectSEX','TargetCODE','TargetNATURE','Time_ms'} <= set(df.columns)
    assert df['SubjectSEX'].dtype == object              # strings, not encoded
    assert set(df['TargetNATURE'].unique()) <= {'R','A'}
    assert set(df['SubjectSEX'].unique()) <= {'M','F'}
    # time axis anchored on trigger row
    one = df[df['SubjectID'] == df['SubjectID'].iloc[0]]
    assert (one['Time_ms'] < 0).any() and (one['Time_ms'] > 0).any()
