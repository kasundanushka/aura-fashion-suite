"""
Prophet Demand Forecasting Model for Clothing AI System.
Implements Facebook Prophet additive model with weekly & yearly seasonality,
holiday/promotional regressors, and uncertainty intervals.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.evaluation.metrics import calculate_forecasting_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ProphetForecaster")

# Check if prophet is importable
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        logger.warning("Prophet not available in current environment. Fallback Bayesian-trend engine will be active.")

def forecast_with_prophet(
    df_sku: pd.DataFrame,
    sku: str,
    horizon_weeks: int = 8,
    test_days: int = 56
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Primary forecasting implementation using Prophet."""
    df_sorted = df_sku.sort_values("date").copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])

    # Prepare Prophet format: ds (datestamp) and y (target)
    prophet_df = pd.DataFrame({
        "ds": df_sorted["date"],
        "y": df_sorted["quantity_sold"].astype(float)
    })
    
    # Extra regressors
    if "discount_applied" in df_sorted.columns:
        prophet_df["discount_applied"] = df_sorted["discount_applied"].fillna(0.0)
    else:
        prophet_df["discount_applied"] = 0.0

    n_samples = len(prophet_df)
    train_idx = n_samples - test_days if n_samples > test_days + 14 else n_samples

    train_df = prophet_df.iloc[:train_idx].copy()
    test_df = prophet_df.iloc[train_idx:].copy()

    # Configure Prophet Model
    model = Prophet(
        growth="linear",
        yearly_seasonality=(len(train_df) >= 730),
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode="additive"
    )
    model.add_regressor("discount_applied")

    # Suppress verbose logging from cmdstanpy / pystan during fit
    import logging as py_logging
    py_logging.getLogger("cmdstanpy").setLevel(py_logging.WARNING)
    py_logging.getLogger("prophet").setLevel(py_logging.WARNING)

    model.fit(train_df)

    # Train predictions & metrics
    train_forecast = model.predict(train_df[["ds", "discount_applied"]])
    train_preds = np.clip(train_forecast["yhat"].values, 0, None)
    train_metrics = calculate_forecasting_metrics(train_df["y"].values, train_preds)

    # Test predictions & metrics
    if not test_df.empty:
        test_forecast = model.predict(test_df[["ds", "discount_applied"]])
        test_preds = np.clip(test_forecast["yhat"].values, 0, None)
        test_metrics = calculate_forecasting_metrics(test_df["y"].values, test_preds)
    else:
        test_metrics = train_metrics

    # Future forecast
    future_days = horizon_weeks * 7
    future_df = model.make_future_dataframe(periods=future_days, freq="D", include_history=False)
    future_df["discount_applied"] = 0.0  # standard baseline for future unless scheduled

    forecast_output = model.predict(future_df)
    
    forecast_records = pd.DataFrame({
        "sku": sku,
        "date": forecast_output["ds"].dt.strftime("%Y-%m-%d"),
        "predicted_quantity": np.round(np.clip(forecast_output["yhat"].values, 0, None), 2),
        "lower_bound": np.round(np.clip(forecast_output["yhat_lower"].values, 0, None), 2),
        "upper_bound": np.round(np.clip(forecast_output["yhat_upper"].values, 0, None), 2),
        "model_used": "Prophet"
    })

    evaluation_report = {
        "model": "Prophet (Additive Seasonality + Regressors)",
        "sku": sku,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "horizon_weeks": horizon_weeks
    }

    return forecast_records, evaluation_report

def forecast_with_fallback(
    df_sku: pd.DataFrame,
    sku: str,
    horizon_weeks: int = 8,
    test_days: int = 56
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust fallback forecaster using scikit-learn Ridge regression + seasonal Fourier terms
    when Prophet package is unavailable.
    """
    from sklearn.linear_model import Ridge
    df_sorted = df_sku.sort_values("date").copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])

    n_samples = len(df_sorted)
    df_sorted["time_step"] = np.arange(n_samples)
    
    feature_cols = [
        "time_step", "sin_month", "cos_month", "sin_dow", "cos_dow",
        "is_weekend", "price_ratio", "discount_applied"
    ]
    X = df_sorted[feature_cols].fillna(0.0)
    y = df_sorted["quantity_sold"]

    train_idx = n_samples - test_days if n_samples > test_days + 14 else n_samples
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_test, y_test = X.iloc[train_idx:], y.iloc[train_idx:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    train_preds = np.clip(model.predict(X_train), 0, None)
    test_preds = np.clip(model.predict(X_test), 0, None) if not X_test.empty else train_preds

    train_metrics = calculate_forecasting_metrics(y_train.values, train_preds)
    test_metrics = calculate_forecasting_metrics(y_test.values, test_preds) if not X_test.empty else train_metrics

    residuals = y_train.values - train_preds
    sigma = np.std(residuals) if len(residuals) > 1 else 1.0

    last_date = df_sorted["date"].max()
    future_days = horizon_weeks * 7
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

    future_records = []
    for idx, f_date in enumerate(future_dates):
        month = f_date.month
        dow = f_date.weekday()
        future_records.append({
            "time_step": n_samples + idx,
            "sin_month": np.sin(2 * np.pi * month / 12.0),
            "cos_month": np.cos(2 * np.pi * month / 12.0),
            "sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "cos_dow": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": 1 if dow in [5, 6] else 0,
            "price_ratio": 1.0,
            "discount_applied": 0.0
        })

    df_future_X = pd.DataFrame(future_records)[feature_cols]
    pred_quantities = np.clip(model.predict(df_future_X), 0, None)
    lower_bounds = np.clip(pred_quantities - 1.96 * sigma, 0, None)
    upper_bounds = pred_quantities + 1.96 * sigma

    forecast_df = pd.DataFrame({
        "sku": sku,
        "date": future_dates.strftime("%Y-%m-%d"),
        "predicted_quantity": np.round(pred_quantities, 2),
        "lower_bound": np.round(lower_bounds, 2),
        "upper_bound": np.round(upper_bounds, 2),
        "model_used": "SeasonalRidgeRegression"
    })

    evaluation_report = {
        "model": "Seasonal Ridge Regression (Fallback)",
        "sku": sku,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "horizon_weeks": horizon_weeks
    }

    return forecast_df, evaluation_report

def forecast_demand(
    sku: str,
    horizon_weeks: int = 8,
    df_features: pd.DataFrame = None,
    prefer_prophet: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified forecasting entrypoint.
    Executes Prophet when available, with automatic failover to robust seasonal regression.
    
    Function signature:
      forecast_demand(sku: str, horizon_weeks: int) -> Tuple[pd.DataFrame, Dict[str, Any]]
    """
    if df_features is None:
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run data pipeline first.")
        df_features = pd.read_csv(feature_path)

    df_sku = df_features[df_features["sku"] == sku].copy()
    if df_sku.empty:
        raise ValueError(f"SKU '{sku}' was not found in the processed sales dataset.")

    if PROPHET_AVAILABLE and prefer_prophet:
        try:
            return forecast_with_prophet(df_sku, sku, horizon_weeks=horizon_weeks)
        except Exception as e:
            logger.error(f"Prophet forecast failed for {sku}: {e}. Falling back to seasonal regression.")
            return forecast_with_fallback(df_sku, sku, horizon_weeks=horizon_weeks)
    else:
        return forecast_with_fallback(df_sku, sku, horizon_weeks=horizon_weeks)

if __name__ == "__main__":
    df_fc, metrics = forecast_demand("OUT-001", horizon_weeks=8)
    print("Prophet / Primary Forecast Sample (Head):")
    print(df_fc.head())
    print("\nEvaluation Metrics:", metrics)
