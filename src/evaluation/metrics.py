"""
Centralised Evaluation Metrics for Time-Series Forecasting and Classification.
Provides standardized, robust metric calculations for models, notebooks, and backend services.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. Regression / Forecasting Metrics
# ==========================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(mean_absolute_error(y_true, y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE) with epsilon smoothing
    to prevent division-by-zero on low/zero sales days.
    Returns percentage as 0-100%.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(round(mape, 2))

def calculate_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes a comprehensive suite of regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2),
        "r2_score": round(r2, 4)
    }

# ==========================================
# 2. Classification Metrics (Trend Direction)
# ==========================================

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> Dict[str, Any]:
    """
    Computes classification metrics including Accuracy, Macro F1, Precision,
    Recall, and Confusion Matrix.
    """
    if labels is None:
        labels = ["rising", "stable", "declining"]

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "confusion_matrix": cm,
        "labels": labels
    }
