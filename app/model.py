"""Model training utilities for linear regression."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a simple Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
