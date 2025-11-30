"""Visualization utilities for regression results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXAMPLES_DIR = Path("examples")


def plot_predictions(
    y_true: pd.Series, y_pred: np.ndarray, filename: Optional[str] = None
) -> Path:
    """Plot predicted vs. actual values and save to disk.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        filename: Optional custom filename for the SVG plot.

    Returns:
        Path to the saved plot.
    """
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    output_name = filename or "predictions_vs_actual.svg"
    output_path = EXAMPLES_DIR / output_name

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Predictions vs Actual")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path
