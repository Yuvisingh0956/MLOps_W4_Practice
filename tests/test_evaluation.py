import pytest
import pandas as pd
import joblib

def test_model_accuracy():
    model = joblib.load('model.pkl')
    df = pd.read_csv('data/data.csv')
    X, y = df.drop('target', axis=1), df['target']
    assert model.score(X, y) > 0.9
