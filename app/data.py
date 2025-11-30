"""Data loading utilities for linear regression examples."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_housing_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset.

    Returns:
        Tuple containing the feature dataframe ``X`` and target series ``y``.
    """
    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y
