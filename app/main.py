"""End-to-end pipeline to train and evaluate Linear Regression."""
from __future__ import annotations

from pathlib import Path

from app.data import load_housing_data
from app.evaluate import evaluate_regression
from app.model import train_linear_regression
from app.preprocess import scale_features, split_data
from app.visualize import plot_predictions


def run_pipeline() -> None:
    """Execute the full training and evaluation workflow."""
    # 1. Load data
    X, y = load_housing_data()

    # 2. Split and scale
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 3. Train model
    model = train_linear_regression(X_train_scaled, y_train)

    # 4. Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_regression(y_test, y_pred)

    # 5. Visualize results
    plot_path = plot_predictions(y_test, y_pred)

    # 6. Report metrics
    print("Linear Regression performance (California Housing dataset):")
    for name, value in metrics.items():
        print(f"- {name.upper()}: {value:.4f}")
    print(f"Plot saved to: {Path(plot_path).resolve()}")


if __name__ == "__main__":
    run_pipeline()
