"""
Baseline Regression Forecasting Model.
Implements an ordinary least squares Linear Regression baseline model using
time index, cyclical calendar harmonics (month, day of week), and promotional pricing.
Provides benchmark metrics (RMSE, MAPE, MAE) to evaluate Prophet improvements.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from src.evaluation.metrics import calculate_forecasting_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BaselineForecaster")

def prepare_regression_features(df_sku: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Extracts regression design matrix X and target y."""
    df_sorted = df_sku.sort_values("date").copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    
    # Feature columns
    feature_cols = [
        "sin_month", "cos_month", "sin_dow", "cos_dow",
        "is_weekend", "price_ratio", "discount_applied"
    ]
    
    # Time step index (integer trend feature: 0, 1, 2, ...)
    df_sorted["time_step"] = np.arange(len(df_sorted))
    feature_cols.append("time_step")

    # Add rolling lag if available
    if "lag_7d" in df_sorted.columns:
        feature_cols.append("lag_7d")

    X = df_sorted[feature_cols].fillna(0.0)
    y = df_sorted["quantity_sold"]

    return df_sorted, X, y, feature_cols

def forecast_baseline(
    sku: str,
    horizon_weeks: int = 8,
    df_features: pd.DataFrame = None,
    test_days: int = 56
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits baseline linear regression for a specific SKU.
    Evaluates on test split (last test_days) and produces future forecast.
    
    Returns:
      forecast_df: DataFrame with ['date', 'predicted_quantity', 'lower_bound', 'upper_bound', 'model_used']
      metrics: Dictionary with train & test RMSE, MAE, MAPE, R2
    """
    if df_features is None:
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run pipeline first.")
        df_features = pd.read_csv(feature_path)

    df_sku = df_features[df_features["sku"] == sku].copy()
    if df_sku.empty:
        raise ValueError(f"SKU '{sku}' not found in feature dataset.")

    df_sorted, X, y, feature_cols = prepare_regression_features(df_sku)
    n_samples = len(df_sorted)

    # Train / Test split
    if n_samples > test_days + 14:
        train_idx = n_samples - test_days
        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_test, y_test = X.iloc[train_idx:], y.iloc[train_idx:]
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Fit Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 0, None)
    test_preds = np.clip(model.predict(X_test), 0, None)

    train_metrics = calculate_forecasting_metrics(y_train.values, train_preds)
    test_metrics = calculate_forecasting_metrics(y_test.values, test_preds)

    # Calculate standard error of residuals for confidence bounds
    residuals = y_train.values - train_preds
    sigma = np.std(residuals) if len(residuals) > 1 else 1.0

    # Generate future horizon dates (daily)
    last_date = df_sorted["date"].max()
    future_days = horizon_weeks * 7
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    future_records = []
    last_lag = df_sorted["quantity_sold"].iloc[-7:].mean() if len(df_sorted) >= 7 else df_sorted["quantity_sold"].mean()

    for idx, f_date in enumerate(future_dates):
        month = f_date.month
        dow = f_date.weekday()
        row_feat = {
            "sin_month": np.sin(2 * np.pi * month / 12.0),
            "cos_month": np.cos(2 * np.pi * month / 12.0),
            "sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "cos_dow": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": 1 if dow in [5, 6] else 0,
            "price_ratio": 1.0,
            "discount_applied": 0.0,
            "time_step": n_samples + idx
        }
        if "lag_7d" in feature_cols:
            row_feat["lag_7d"] = last_lag

        future_records.append(row_feat)

    df_future_X = pd.DataFrame(future_records)[feature_cols]
    raw_future_preds = model.predict(df_future_X)
    pred_quantities = np.clip(raw_future_preds, 0, None)

    # 95% prediction interval (z = 1.96)
    lower_bounds = np.clip(pred_quantities - 1.96 * sigma, 0, None)
    upper_bounds = pred_quantities + 1.96 * sigma

    forecast_df = pd.DataFrame({
        "sku": sku,
        "date": future_dates.strftime("%Y-%m-%d"),
        "predicted_quantity": np.round(pred_quantities, 2),
        "lower_bound": np.round(lower_bounds, 2),
        "upper_bound": np.round(upper_bounds, 2),
        "model_used": "BaselineLinearRegression"
    })

    evaluation_report = {
        "model": "Baseline Linear Regression",
        "sku": sku,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coefficients": dict(zip(feature_cols, [round(float(c), 4) for c in model.coef_])),
        "intercept": round(float(model.intercept_), 4)
    }

    return forecast_df, evaluation_report

if __name__ == "__main__":
    df_fc, rep = forecast_baseline("OUT-001", horizon_weeks=8)
    print("Baseline Forecast Sample (Head):")
    print(df_fc.head())
    print("\nEvaluation Report:", rep["test_metrics"])
