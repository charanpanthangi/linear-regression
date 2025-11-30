# Linear Regression Tutorial & Template

A beginner-friendly, end-to-end example of training, evaluating, and visualizing a **Linear Regression** model using scikit-learn. This repo doubles as both a mini-tutorial and a starting template for your own regression projects.

## Why Linear Regression?
Linear Regression models the relationship between one or more input variables and a continuous target. It is a great first model for:
- Predicting continuous outcomes (prices, scores, measurements).
- Establishing a simple baseline before trying more complex models.
- Interpreting feature influence through coefficients.

**Limitations:**
- Assumes a linear relationship between features and the target.
- Sensitive to outliers and multicollinearity.
- May underfit complex, nonlinear patterns.

## Dataset
The examples use the **California Housing** dataset from scikit-learn. The task is to predict median house value from demographic and geographic features.

## How the Pipeline Works
1. **Data loading (`app/data.py`)** – Fetch the dataset and return features `X` and target `y`.
2. **Preprocessing (`app/preprocess.py`)** – Split into train/test sets and scale numeric features with `StandardScaler`.
3. **Model training (`app/model.py`)** – Fit a simple `LinearRegression` model.
4. **Evaluation (`app/evaluate.py`)** – Compute MSE, MAE, RMSE, and R².
5. **Visualization (`app/visualize.py`)** – Plot predicted vs. actual values as an SVG saved to `examples/`.
6. **Orchestration (`app/main.py`)** – Run the full pipeline and print metrics.

## Project Structure
```
├── app/
│   ├── data.py          # Load dataset
│   ├── preprocess.py    # Split and scale data
│   ├── model.py         # Train Linear Regression
│   ├── evaluate.py      # Compute metrics
│   ├── visualize.py     # Save prediction plot
│   └── main.py          # Run the full workflow
├── notebooks/
│   └── demo_linear_regression.ipynb
├── tests/               # Basic pytest checks
├── examples/            # Generated plots
├── requirements.txt
├── Dockerfile
├── LICENSE (MIT)
└── README.md
```

## Getting Started
### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python app/main.py
```
This prints evaluation metrics and saves `examples/predictions_vs_actual.svg`.

### 3. Explore the notebook
```bash
jupyter notebook notebooks/demo_linear_regression.ipynb
```
The notebook walks through the same steps with explanations and visuals.

### 4. Run tests
```bash
pytest
```

## When to Use Linear Regression
- You need a quick, interpretable baseline.
- Relationships between inputs and target are roughly linear.
- You want to understand feature influence via coefficients.

## Future Ideas
- Add regularization (Ridge/Lasso).
- Feature engineering for non-linear relationships.
- Hyperparameter search with cross-validation.
- Logging and experiment tracking.

Happy learning!
