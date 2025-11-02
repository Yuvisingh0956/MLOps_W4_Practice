import pytest
import pandas as pd

def test_data_shape():
    df = pd.read_csv('data/data.csv')
    assert df.shape[1] == 5  # 4 features + 1 target
