import numpy as np

from app.data import load_housing_data
from app.model import train_linear_regression
from app.preprocess import scale_features, split_data


def test_model_can_predict():
    X, y = load_housing_data()
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.2, random_state=0)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    model = train_linear_regression(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test_scaled.shape[0]
