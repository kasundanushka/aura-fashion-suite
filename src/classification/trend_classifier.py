"""
Random Forest Trend Direction Classifier for Clothing AI System.
Predicts whether a fashion product is currently 'rising', 'stable', or 'declining'
using time-series momentum, rolling volatility, category context, and seasonal features.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from src.evaluation.metrics import calculate_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrendClassifier")

CLASSIFICATION_FEATURE_COLS = [
    "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_28d",
    "rolling_std_7d", "rolling_std_14d", "rolling_std_28d",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "sales_growth_rate", "days_since_launch", "price_ratio",
    "discount_applied", "category_daily_avg_demand",
    "sin_month", "cos_month", "sin_dow", "cos_dow"
]

class FashionTrendModel:
    """Encapsulates training, inference, and feature importance for the Random Forest Trend Classifier."""
    
    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_cols = []
        self.is_fitted = False
        self.classes_ = []
        self.evaluation_report = {}

    def fit_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Trains the Random Forest model and logs train/test performance."""
        self.feature_cols = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
        df_clean = df.dropna(subset=["trend_label"]).copy()
        df_clean[self.feature_cols] = df_clean[self.feature_cols].fillna(0.0)
        
        # Chronological split
        df_clean = df_clean.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df_clean) * (1 - test_size))

        X_train = df_clean.iloc[:split_idx][self.feature_cols]
        y_train = df_clean.iloc[:split_idx]["trend_label"]
        X_test = df_clean.iloc[split_idx:][self.feature_cols]
        y_test = df_clean.iloc[split_idx:]["trend_label"]

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.classes_ = list(self.model.classes_)

        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        train_metrics = calculate_classification_metrics(y_train.values, train_preds, labels=self.classes_)
        test_metrics = calculate_classification_metrics(y_test.values, test_preds, labels=self.classes_)

        # Extract top feature importances
        importances = self.model.feature_importances_
        feature_importance_list = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
        ]

        self.evaluation_report = {
            "model": "Random Forest Classifier (Primary)",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance_list
        }

        logger.info(f"RandomForest Fit Complete. Test Accuracy: {test_metrics['accuracy']}, Macro F1: {test_metrics['f1_macro']}")
        return self.evaluation_report

    def predict_sku_trend(self, df_sku_latest: pd.DataFrame) -> Dict[str, Any]:
        """Predicts trend status and probability distribution for a single SKU record."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call fit_and_evaluate first.")

        X = df_sku_latest[self.feature_cols].fillna(0.0)
        pred_label = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, probs)}
        conf_score = round(float(np.max(probs)), 4)

        return {
            "trend_status": pred_label,
            "confidence_score": conf_score,
            "probabilities": prob_dict
        }

# Global singleton instance for memory efficiency & fast serving
_GLOBAL_CLASSIFIER = None

def reset_trend_classifier():
    """Resets the singleton classifier so that newly ingested data triggers re-fitting."""
    global _GLOBAL_CLASSIFIER
    _GLOBAL_CLASSIFIER = None
    logger.info("Reset trend classifier cache. Next prediction will fit against latest feature matrix.")

def get_trend_classifier(df_features: pd.DataFrame = None) -> FashionTrendModel:
    """Returns singleton instance of trained FashionTrendModel."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None or not _GLOBAL_CLASSIFIER.is_fitted:
        if df_features is None:
            feature_path = "data/processed/feature_matrix.csv"
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run data pipeline first.")
            df_features = pd.read_csv(feature_path)

        classifier = FashionTrendModel()
        classifier.fit_and_evaluate(df_features)
        _GLOBAL_CLASSIFIER = classifier

    return _GLOBAL_CLASSIFIER

def classify_trend(sku: str, df_features: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Unified trend classification entrypoint.
    Returns trend status ('rising', 'stable', 'declining') and confidence metrics.
    
    Function signature:
      classify_trend(sku: str) -> dict
    """
    if df_features is None:
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run data pipeline first.")
        df_features = pd.read_csv(feature_path)

    df_sku = df_features[df_features["sku"] == sku].sort_values("date")
    if df_sku.empty:
        raise ValueError(f"SKU '{sku}' not found in feature dataset.")

    latest_row = df_sku.iloc[[-1]]
    classifier = get_trend_classifier(df_features)
    res = classifier.predict_sku_trend(latest_row)

    return {
        "sku": sku,
        "trend_status": res["trend_status"],
        "confidence_score": res["confidence_score"],
        "probabilities": res["probabilities"],
        "model_used": "RandomForestClassifier",
        "growth_rate_recent": round(float(latest_row["sales_growth_rate"].values[0]), 4),
        "latest_daily_mean_28d": round(float(latest_row["rolling_mean_28d"].values[0]), 2),
        "feature_importance": classifier.evaluation_report.get("feature_importance", [])[:5]
    }

if __name__ == "__main__":
    rep = classify_trend("OUT-001")
    print("Trend Classification Output for OUT-001:")
    print(rep)
