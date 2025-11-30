import numpy as np

from app.evaluate import evaluate_regression


def test_evaluate_returns_numeric_metrics():
    y_true = np.array([3.0, 4.0, 5.0])
    y_pred = np.array([2.5, 4.5, 5.5])
    metrics = evaluate_regression(y_true, y_pred)

    expected_keys = {"mse", "mae", "rmse", "r2"}
    assert set(metrics.keys()) == expected_keys
    for value in metrics.values():
        assert isinstance(value, float)
