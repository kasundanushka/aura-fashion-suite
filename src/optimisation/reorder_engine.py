"""
Inventory Reorder Optimization Engine for Clothing AI System.
Combines Time-Series Demand Forecasts, Supervised Trend Classifications,
Safety Stock Buffers, and Current Stock to generate automated replenishment recommendations.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.forecasting.prophet_model import forecast_demand
from src.classification.trend_classifier import classify_trend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ReorderOptimizer")

# Default Hyperparameters
DEFAULT_SERVICE_FACTOR_Z = 1.65  # 95% cycle service level
DEFAULT_LEAD_TIME_DAYS = 7        # 1 week supplier lead time
DEFAULT_REVIEW_HORIZON_WEEKS = 4  # 4-week order cycle planning horizon

TREND_DEMAND_MULTIPLIERS = {
    "rising": 1.15,     # +15% proactive buffer for trending products
    "stable": 1.00,     # Standard replenishment
    "declining": 0.80   # -20% conservative curtailment to mitigate deadstock risk
}

def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    service_factor_z: float = DEFAULT_SERVICE_FACTOR_Z
) -> int:
    """
    Calculates statistical safety stock:
      Safety Stock = ceil(Z * sigma_daily * sqrt(Lead_Time_Days))
    """
    ss = service_factor_z * float(daily_demand_std) * np.sqrt(lead_time_days)
    return int(np.ceil(max(1, ss)))

def calculate_reorder_recommendation(
    sku: str,
    current_stock: int,
    horizon_weeks: int = DEFAULT_REVIEW_HORIZON_WEEKS,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    service_factor_z: float = DEFAULT_SERVICE_FACTOR_Z,
    df_features: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Computes an end-to-end inventory recommendation for a single SKU by synthesizing:
      1. Demand Forecast (Prophet / Regression)
      2. Trend Classification (Random Forest)
      3. Statistical Safety Stock and Reorder Point
      4. Net Required Order Quantity and Alert Classification
    """
    if df_features is None:
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run pipeline first.")
        df_features = pd.read_csv(feature_path)

    df_sku = df_features[df_features["sku"] == sku].sort_values("date")
    if df_sku.empty:
        raise ValueError(f"SKU '{sku}' not found in feature dataset.")

    latest_row = df_sku.iloc[-1]
    
    # 1. Get Demand Forecast
    df_forecast, fc_eval = forecast_demand(sku, horizon_weeks=horizon_weeks, df_features=df_features)
    total_forecasted_demand = float(df_forecast["predicted_quantity"].sum())
    daily_mean_forecast = total_forecasted_demand / (horizon_weeks * 7)

    # 2. Get Trend Classification
    trend_info = classify_trend(sku, df_features=df_features)
    trend_status = trend_info["trend_status"]
    trend_multiplier = TREND_DEMAND_MULTIPLIERS.get(trend_status, 1.00)

    # 3. Calculate Safety Stock & Reorder Point
    daily_std = float(latest_row.get("rolling_std_28d", latest_row.get("rolling_std_14d", 2.0)))
    safety_stock = calculate_safety_stock(
        daily_demand_std=daily_std,
        lead_time_days=lead_time_days,
        service_factor_z=service_factor_z
    )

    lead_time_demand = daily_mean_forecast * lead_time_days
    reorder_point = int(np.ceil(lead_time_demand + safety_stock))

    # 4. Adjusted Target Demand & Net Reorder Quantity
    adjusted_horizon_demand = total_forecasted_demand * trend_multiplier
    target_inventory_level = adjusted_horizon_demand + safety_stock
    net_recommended_order = int(np.ceil(max(0, target_inventory_level - current_stock)))

    # 5. Alert Level Classification Logic
    # - URGENT_REORDER: Current stock is below or equal to 50% safety stock or runs out during lead time
    # - REORDER_SOON: Current stock <= Reorder Point (ROP)
    # - OVERSTOCKED: Current stock > 2.5x total horizon target
    # - SUFFICIENT_STOCK: Stock is healthy
    if current_stock <= max(2, int(0.5 * safety_stock)) or current_stock < lead_time_demand:
        alert_level = "URGENT_REORDER"
        action_summary = f"Critically low inventory! Stock ({current_stock}) is below safety threshold ({safety_stock}). Reorder {net_recommended_order} units immediately."
    elif current_stock <= reorder_point:
        alert_level = "REORDER_SOON"
        action_summary = f"Stock level ({current_stock}) has breached Reorder Point ({reorder_point}). Place order for {net_recommended_order} units."
    elif current_stock > 2.5 * (adjusted_horizon_demand + safety_stock):
        alert_level = "OVERSTOCKED"
        action_summary = f"Excess inventory detected ({current_stock} units). Hold orders; consider promotional discounting."
        net_recommended_order = 0
    else:
        alert_level = "SUFFICIENT_STOCK"
        action_summary = f"Stock level ({current_stock}) is optimal for forecasted demand over next {horizon_weeks} weeks."
        net_recommended_order = 0

    return {
        "sku": sku,
        "current_stock": int(current_stock),
        "alert_level": alert_level,
        "recommended_quantity": net_recommended_order,
        "forecasted_demand_total": round(total_forecasted_demand, 1),
        "adjusted_forecast_demand": round(adjusted_horizon_demand, 1),
        "daily_mean_forecast": round(daily_mean_forecast, 2),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "trend_status": trend_status,
        "trend_multiplier": trend_multiplier,
        "trend_confidence": trend_info["confidence_score"],
        "action_summary": action_summary,
        "horizon_weeks": horizon_weeks,
        "lead_time_days": lead_time_days
    }

def generate_all_inventory_alerts(
    df_stock: pd.DataFrame = None,
    df_features: pd.DataFrame = None
) -> List[Dict[str, Any]]:
    """Generates inventory reorder recommendations across all active SKUs in the catalog."""
    if df_stock is None:
        stock_file = "data/raw/stock_levels.csv"
        if not os.path.exists(stock_file):
            raise FileNotFoundError(f"Stock levels file not found at {stock_file}.")
        df_stock = pd.read_csv(stock_file)

    if df_features is None:
        feature_path = "data/processed/feature_matrix.csv"
        df_features = pd.read_csv(feature_path)

    recommendations = []
    for _, row in df_stock.iterrows():
        sku = str(row["sku"]).strip()
        stock = int(row["current_stock"])
        try:
            rec = calculate_reorder_recommendation(
                sku=sku,
                current_stock=stock,
                df_features=df_features
            )
            recommendations.append(rec)
        except Exception as e:
            logger.error(f"Error calculating reorder recommendation for {sku}: {e}")

    return recommendations

if __name__ == "__main__":
    rec = calculate_reorder_recommendation("OUT-001", current_stock=45)
    print("Optimization Recommendation for OUT-001:")
    for k, v in rec.items():
        print(f"  {k}: {v}")
