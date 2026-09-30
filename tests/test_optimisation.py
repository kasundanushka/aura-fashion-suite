"""
Unit Tests for Inventory Reorder Optimization Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.optimisation.reorder_engine import (
    calculate_safety_stock,
    calculate_reorder_recommendation,
    TREND_DEMAND_MULTIPLIERS
)

def test_safety_stock_formula():
    # SS = Z * sigma * sqrt(L) = 1.65 * 4.0 * sqrt(9) = 1.65 * 4 * 3 = 19.8 -> ceil(19.8) = 20
    ss = calculate_safety_stock(daily_demand_std=4.0, lead_time_days=9, service_factor_z=1.65)
    assert ss == 20

def test_trend_multipliers_values():
    assert TREND_DEMAND_MULTIPLIERS["rising"] == 1.15
    assert TREND_DEMAND_MULTIPLIERS["stable"] == 1.00
    assert TREND_DEMAND_MULTIPLIERS["declining"] == 0.80

def test_reorder_recommendation_urgent_alert():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "sku": ["OUT-URGENT"] * 60,
        "date": dates,
        "quantity_sold": [20] * 60,
        "rolling_mean_7d": [20.0] * 60,
        "rolling_mean_14d": [20.0] * 60,
        "rolling_mean_28d": [20.0] * 60,
        "rolling_std_7d": [2.0] * 60,
        "rolling_std_14d": [2.0] * 60,
        "rolling_std_28d": [2.0] * 60,
        "lag_1d": [20.0] * 60,
        "lag_7d": [20.0] * 60,
        "lag_14d": [20.0] * 60,
        "lag_28d": [20.0] * 60,
        "sales_growth_rate": [0.0] * 60,
        "days_since_launch": list(range(60)),
        "price_ratio": [1.0] * 60,
        "discount_applied": [0.0] * 60,
        "category_daily_avg_demand": [20.0] * 60,
        "sin_month": [0.5] * 60,
        "cos_month": [0.5] * 60,
        "sin_dow": [0.5] * 60,
        "cos_dow": [0.5] * 60,
        "trend_label": ["stable"] * 60
    })

    # Critically low stock (current_stock=2) with daily demand ~20
    rec = calculate_reorder_recommendation(
        sku="OUT-URGENT",
        current_stock=2,
        horizon_weeks=4,
        lead_time_days=7,
        df_features=df
    )

    assert rec["alert_level"] == "URGENT_REORDER"
    assert rec["recommended_quantity"] > 0
    assert "Critically low" in rec["action_summary"]

def test_reorder_recommendation_overstocked():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "sku": ["OUT-EXCESS"] * 60,
        "date": dates,
        "quantity_sold": [5] * 60,
        "rolling_mean_7d": [5.0] * 60,
        "rolling_mean_14d": [5.0] * 60,
        "rolling_mean_28d": [5.0] * 60,
        "rolling_std_7d": [1.0] * 60,
        "rolling_std_14d": [1.0] * 60,
        "rolling_std_28d": [1.0] * 60,
        "lag_1d": [5.0] * 60,
        "lag_7d": [5.0] * 60,
        "lag_14d": [5.0] * 60,
        "lag_28d": [5.0] * 60,
        "sales_growth_rate": [0.0] * 60,
        "days_since_launch": list(range(60)),
        "price_ratio": [1.0] * 60,
        "discount_applied": [0.0] * 60,
        "category_daily_avg_demand": [5.0] * 60,
        "sin_month": [0.5] * 60,
        "cos_month": [0.5] * 60,
        "sin_dow": [0.5] * 60,
        "cos_dow": [0.5] * 60,
        "trend_label": ["stable"] * 60
    })

    # Huge current stock (2000 units) for small demand
    rec = calculate_reorder_recommendation(
        sku="OUT-EXCESS",
        current_stock=2000,
        horizon_weeks=4,
        lead_time_days=7,
        df_features=df
    )

    assert rec["alert_level"] == "OVERSTOCKED"
    assert rec["recommended_quantity"] == 0
