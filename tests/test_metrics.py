import numpy as np

from roughness.metrics import regression_metrics


def test_regression_metrics_exact_prediction():
    y = np.array([1.0, 2.0, 3.0])

    result = regression_metrics(y, y, np.array([0.5, 0.25, 0.25]))

    assert result == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_regression_metrics_honor_sample_weight():
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 2.0])

    weighted = regression_metrics(y_true, y_pred, np.array([0.1, 0.9]))

    assert weighted["mae"] == 0.1
