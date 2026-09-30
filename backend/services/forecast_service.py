"""
Forecast Service for Flask Backend.
Handles retrieval of historical sales, demand forecasts (Prophet + Baseline comparison),
in-memory response caching, and persisting forecasts to SQLite database.
"""

import os
import sys
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.init_db import get_db_connection
from src.forecasting.prophet_model import forecast_demand
from src.forecasting.baseline_regression import forecast_baseline

logger = logging.getLogger("ForecastService")

# In-memory forecast cache: (sku, weeks, include_baseline) -> (timestamp, result_dict)
_FORECAST_CACHE: Dict[tuple, tuple] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes

def clear_forecast_cache():
    """Clears cached forecast results whenever a new dataset is uploaded or models retrain."""
    global _FORECAST_CACHE
    _FORECAST_CACHE.clear()
    logger.info("Cleared forecast memory cache.")

class ForecastService:
    """Service layer managing time series demand forecasting queries with caching and persistence."""

    @staticmethod
    def get_forecast_for_sku(sku: str, weeks: int = 8, include_baseline: bool = True) -> Dict[str, Any]:
        """
        Retrieves historical sales, generates future forecast (with confidence interval),
        and provides model performance evaluation. Uses in-memory TTL cache for high performance.
        """
        cache_key = (sku, weeks, include_baseline)
        now = time.time()
        if cache_key in _FORECAST_CACHE:
            cached_time, cached_result = _FORECAST_CACHE[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                return cached_result

        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError("Feature matrix not found. Data pipeline must be executed first.")

        df_features = pd.read_csv(feature_path)
        df_sku = df_features[df_features["sku"] == sku].sort_values("date")

        if df_sku.empty:
            raise ValueError(f"SKU '{sku}' not found.")

        # Historical series (last 90 days for UI chart visualization)
        hist_df = df_sku.iloc[-90:][["date", "quantity_sold", "unit_price", "discount_applied"]].copy()
        historical_data = hist_df.to_dict(orient="records")

        # Primary Forecast (Prophet / Fallback)
        forecast_df, eval_report = forecast_demand(sku, horizon_weeks=weeks, df_features=df_features)
        forecast_data = forecast_df.to_dict(orient="records")

        # Baseline Comparison (Linear Regression)
        baseline_data = None
        baseline_report = None
        if include_baseline:
            try:
                base_df, base_eval = forecast_baseline(sku, horizon_weeks=weeks, df_features=df_features)
                baseline_data = base_df.to_dict(orient="records")
                baseline_report = base_eval
            except Exception as e:
                logger.debug(f"Baseline forecast error for {sku}: {e}")

        # Save latest forecast points to database (delete stale points for this SKU first to prevent bloat)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM forecasts WHERE sku = ?", (sku,))
            for row in forecast_data:
                cursor.execute(
                    """
                    INSERT INTO forecasts (sku, forecast_date, predicted_quantity, lower_bound, upper_bound, model_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (sku, row["date"], row["predicted_quantity"], row["lower_bound"], row["upper_bound"], row["model_used"])
                )
            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning(f"Could not persist forecasts for {sku}: {err}")

        result = {
            "sku": sku,
            "horizon_weeks": weeks,
            "historical": historical_data,
            "forecast": forecast_data,
            "baseline": baseline_data,
            "model_evaluation": eval_report,
            "baseline_evaluation": baseline_report,
            "summary": {
                "total_predicted_demand": round(float(forecast_df["predicted_quantity"].sum()), 1),
                "weekly_average_demand": round(float(forecast_df["predicted_quantity"].sum() / weeks), 1),
                "peak_forecast_date": str(forecast_df.loc[forecast_df["predicted_quantity"].idxmax()]["date"]),
                "peak_forecast_value": float(forecast_df["predicted_quantity"].max())
            }
        }

        # Cache the result
        _FORECAST_CACHE[cache_key] = (now, result)
        return result
