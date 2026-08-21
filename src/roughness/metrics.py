import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, sample_weight=None) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(
            mean_absolute_error(
                y_true_array, y_pred_array, sample_weight=sample_weight
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_true_array, y_pred_array, sample_weight=sample_weight
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(y_true_array, y_pred_array, sample_weight=sample_weight)
        ),
    }
