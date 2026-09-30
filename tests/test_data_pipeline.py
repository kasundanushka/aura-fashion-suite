"""
Unit Tests for Data Pipeline & Feature Engineering Modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_pipeline.clean_data import (
    validate_sales_data,
    validate_product_data,
    fill_missing_dates_per_sku
)
from src.data_pipeline.feature_engineering import (
    compute_rolling_features,
    compute_temporal_features,
    assign_trend_classification_labels
)

def test_validate_sales_data_valid():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "sku": ["TEST-01", "TEST-01"],
        "quantity_sold": [10, 15],
        "unit_price": [29.99, 29.99],
        "discount_applied": [0.0, 0.1]
    })
    clean = validate_sales_data(raw)
    assert len(clean) == 2
    assert clean["quantity_sold"].iloc[0] == 10
    assert clean["unit_price"].iloc[0] == 29.99

def test_validate_sales_data_rejects_negative_quantities():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "sku": ["TEST-01", "TEST-01"],
        "quantity_sold": [-5, 12],
        "unit_price": [19.99, 19.99]
    })
    clean = validate_sales_data(raw)
    assert len(clean) == 1
    assert clean["quantity_sold"].iloc[0] == 12

def test_fill_missing_dates_per_sku():
    sales = pd.DataFrame({
        "date": [pd.to_datetime("2024-01-01"), pd.to_datetime("2024-01-04")],
        "sku": ["TEST-01", "TEST-01"],
        "quantity_sold": [10, 20],
        "unit_price": [15.0, 15.0],
        "discount_applied": [0.0, 0.0]
    })
    aligned = fill_missing_dates_per_sku(sales)
    # Expected 4 continuous days: 1st, 2nd, 3rd, 4th
    assert len(aligned) == 4
    # Missing 2nd and 3rd days should have quantity_sold = 0
    zero_days = aligned[aligned["quantity_sold"] == 0]
    assert len(zero_days) == 2

def test_temporal_feature_generation():
    df = pd.DataFrame({
        "date": ["2024-06-15"],
        "sku": ["TEST-01"],
        "quantity_sold": [10]
    })
    featured = compute_temporal_features(df)
    assert "day_of_week" in featured.columns
    assert "sin_month" in featured.columns
    assert "cos_month" in featured.columns
    assert featured["month"].iloc[0] == 6

def test_trend_label_assignment():
    # Build 60 days of strong growth
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sales = [5] * 30 + [25] * 30  # +400% surge
    df = pd.DataFrame({
        "date": dates,
        "sku": ["TEST-GROWTH"] * 60,
        "quantity_sold": sales
    })
    labeled = assign_trend_classification_labels(df)
    latest_label = labeled["trend_label"].iloc[-1]
    assert latest_label == "rising"
