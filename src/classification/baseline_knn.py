"""
Baseline K-Nearest Neighbors (KNN) Fashion Trend Classifier.
Serves as a non-parametric distance-based baseline to benchmark RandomForest performance.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import calculate_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BaselineKNNClassifier")

CLASSIFICATION_FEATURE_COLS = [
    "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_28d",
    "rolling_std_7d", "rolling_std_14d", "rolling_std_28d",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "sales_growth_rate", "days_since_launch", "price_ratio",
    "discount_applied", "category_daily_avg_demand",
    "sin_month", "cos_month", "sin_dow", "cos_dow"
]

def train_and_evaluate_knn(
    df: pd.DataFrame = None,
    n_neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[KNeighborsClassifier, StandardScaler, Dict[str, Any]]:
    """Trains KNN classifier on temporal train/test split and computes evaluation metrics."""
    if df is None:
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run data pipeline first.")
        df = pd.read_csv(feature_path)

    # Filter available features
    avail_cols = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
    df_clean = df.dropna(subset=avail_cols + ["trend_label"]).copy()
    
    # Sort chronologically for time-aware split
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_clean) * (1 - test_size))

    X_train = df_clean.iloc[:split_idx][avail_cols]
    y_train = df_clean.iloc[:split_idx]["trend_label"]
    X_test = df_clean.iloc[split_idx:][avail_cols]
    y_test = df_clean.iloc[split_idx:][trend_label_col if "trend_label_col" in locals() else "trend_label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X_train_scaled, y_train)

    train_preds = knn.predict(X_train_scaled)
    test_preds = knn.predict(X_test_scaled)

    train_metrics = calculate_classification_metrics(y_train.values, train_preds)
    test_metrics = calculate_classification_metrics(y_test.values, test_preds)

    report = {
        "model": f"KNN (k={n_neighbors}, weights='distance')",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_count": len(avail_cols)
    }

    return knn, scaler, report

if __name__ == "__main__":
    knn, scaler, report = train_and_evaluate_knn()
    print("Baseline KNN Evaluation Report:")
    print("Test Accuracy:", report["test_metrics"]["accuracy"])
    print("Test Macro F1:", report["test_metrics"]["f1_macro"])
    print("Confusion Matrix:\n", report["test_metrics"]["confusion_matrix"])
