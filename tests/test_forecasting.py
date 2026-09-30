"""
Unit Tests for Time-Series Demand Forecasting and Metrics.
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_forecasting_metrics
)
from src.forecasting.baseline_regression import forecast_baseline
from src.forecasting.prophet_model import forecast_demand

def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])

    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    assert round(mae, 2) == 2.25
    assert rmse > 0
    assert mape > 0

def test_baseline_regression_forecasting():
    # Synthetic test series
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "sku": ["OUT-001"] * 100,
        "date": dates,
        "quantity_sold": np.random.poisson(15, 100),
        "sin_month": np.sin(2 * np.pi * dates.month / 12),
        "cos_month": np.cos(2 * np.pi * dates.month / 12),
        "sin_dow": np.sin(2 * np.pi * dates.dayofweek / 7),
        "cos_dow": np.cos(2 * np.pi * dates.dayofweek / 7),
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
        "price_ratio": 1.0,
        "discount_applied": 0.0
    })

    forecast_df, rep = forecast_baseline("OUT-001", horizon_weeks=4, df_features=df)
    assert len(forecast_df) == 28 # 4 weeks * 7 days
    assert "predicted_quantity" in forecast_df.columns
    assert "lower_bound" in forecast_df.columns
    assert "upper_bound" in forecast_df.columns
    assert (forecast_df["predicted_quantity"] >= 0).all()
    assert (forecast_df["upper_bound"] >= forecast_df["lower_bound"]).all()

def test_prophet_or_fallback_forecasting():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "sku": ["OUT-001"] * 100,
        "date": dates,
        "quantity_sold": np.random.poisson(15, 100),
        "sin_month": np.sin(2 * np.pi * dates.month / 12),
        "cos_month": np.cos(2 * np.pi * dates.month / 12),
        "sin_dow": np.sin(2 * np.pi * dates.dayofweek / 7),
        "cos_dow": np.cos(2 * np.pi * dates.dayofweek / 7),
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
        "price_ratio": 1.0,
        "discount_applied": 0.0
    })

    forecast_df, rep = forecast_demand("OUT-001", horizon_weeks=4, df_features=df)
    assert len(forecast_df) == 28
    assert (forecast_df["predicted_quantity"] >= 0).all()
    assert rep["sku"] == "OUT-001"
