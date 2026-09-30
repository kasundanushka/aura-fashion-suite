"""
Unit Tests for Trend Classification Modules (RandomForest & KNN).
"""

import pytest
import pandas as pd
import numpy as np

from src.classification.baseline_knn import train_and_evaluate_knn
from src.classification.trend_classifier import FashionTrendModel, classify_trend

@pytest.fixture
def sample_feature_matrix():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({
        "sku": ["OUT-001"] * 60 + ["TOP-002"] * 60,
        "date": list(dates[:60]) + list(dates[:60]),
        "quantity_sold": [10] * 60 + [25] * 60,
        "rolling_mean_7d": [10.0] * 60 + [25.0] * 60,
        "rolling_mean_14d": [10.0] * 60 + [25.0] * 60,
        "rolling_mean_28d": [10.0] * 60 + [25.0] * 60,
        "rolling_std_7d": [1.0] * 120,
        "rolling_std_14d": [1.0] * 120,
        "rolling_std_28d": [1.0] * 120,
        "lag_1d": [10.0] * 60 + [25.0] * 60,
        "lag_7d": [10.0] * 60 + [25.0] * 60,
        "lag_14d": [10.0] * 60 + [25.0] * 60,
        "lag_28d": [10.0] * 60 + [25.0] * 60,
        "sales_growth_rate": [0.0] * 60 + [0.25] * 60,
        "days_since_launch": list(range(60)) + list(range(60)),
        "price_ratio": [1.0] * 120,
        "discount_applied": [0.0] * 120,
        "category_daily_avg_demand": [15.0] * 120,
        "sin_month": [0.5] * 120,
        "cos_month": [0.5] * 120,
        "sin_dow": [0.5] * 120,
        "cos_dow": [0.5] * 120,
        "trend_label": ["stable"] * 60 + ["rising"] * 60
    })
    return df

def test_knn_classifier_train_and_eval(sample_feature_matrix):
    knn, scaler, report = train_and_evaluate_knn(sample_feature_matrix, n_neighbors=3, test_size=0.2)
    assert report["train_metrics"]["accuracy"] >= 0.80
    assert "f1_macro" in report["test_metrics"]
    assert len(report["test_metrics"]["confusion_matrix"]) > 0

def test_random_forest_fit_and_inference(sample_feature_matrix):
    model = FashionTrendModel(n_estimators=20, max_depth=5)
    report = model.fit_and_evaluate(sample_feature_matrix, test_size=0.2)
    
    assert model.is_fitted
    assert report["train_metrics"]["accuracy"] >= 0.80
    assert len(report["feature_importance"]) > 0

    latest_row = sample_feature_matrix.iloc[[-1]]
    pred = model.predict_sku_trend(latest_row)
    assert pred["trend_status"] in ["rising", "stable", "declining"]
    assert 0.0 <= pred["confidence_score"] <= 1.0

def test_classify_trend_function(sample_feature_matrix):
    result = classify_trend("TOP-002", df_features=sample_feature_matrix)
    assert result["sku"] == "TOP-002"
    assert result["trend_status"] in ["rising", "stable", "declining"]
    assert result["confidence_score"] > 0
