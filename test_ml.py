import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics

def test_train_model_returns_correct_type():
    """
    Test that train_model returns a RandomForestClassifier instance
    """
    X_test = np.array([[1, 2], [3, 4]])
    y_test = np.array([0, 1])
    model = train_model(X_test, y_test)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_returns_three_floats():
    """
    Test that compute_model_metrics returns three float values
    """
    y_test = np.array([1, 0])
    y_pred_test = np.array([1, 1])
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred_test)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_training_set_size():
    """
    Test that the data split creates a train dataset of the expected size
    """
    test_data = pd.DataFrame({
        "age": [54, 12, 33, 28, 65, 19, 88, 22, 66, 99],
        "salary": ["<=50K"] * 10
    })
    test_train, _ = train_test_split(test_data, test_size=0.20, random_state=42)
    assert len(test_train) == 8
