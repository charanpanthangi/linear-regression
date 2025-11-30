import pandas as pd

from app.data import load_housing_data


def test_load_housing_data_shapes():
    X, y = load_housing_data()
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
