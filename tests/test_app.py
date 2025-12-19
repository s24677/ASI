import pytest
import pandas as pd
import numpy as np
import joblib
import os
from train_model import predict

def test_predict_returns_valid_id():
    from sklearn.linear_model import LinearRegression
    dummy_model = LinearRegression()

    X = np.array([[30, 0], [60, 2]])
    y = np.array([0, 4])
    dummy_model.fit(X, y)
    joblib.dump(dummy_model, 'model.joblib')

    sample_input = np.array([[45, 1]])
    result = predict(sample_input)
    assert isinstance(result, int)
    assert 0 <= result <= 4

    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
