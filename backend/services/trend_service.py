"""
Trend Service for Flask Backend.
Provides trend classification status, confidence probabilities, and historical growth metrics.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from database.init_db import get_db_connection
from src.classification.trend_classifier import classify_trend
from src.classification.baseline_knn import train_and_evaluate_knn

class TrendService:
    """Service layer managing fashion trend classification requests."""

    @staticmethod
    def get_trend_for_sku(sku: str) -> Dict[str, Any]:
        """Classifies the trend direction for a specific SKU."""
        feature_path = "data/processed/feature_matrix.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError("Feature matrix not found. Run pipeline first.")

        df_features = pd.read_csv(feature_path)
        result = classify_trend(sku, df_features=df_features)

        # Retrieve product details
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE sku = ?", (sku,))
        prod_row = cursor.fetchone()

        # Save to trend_labels table (clear older classification for this SKU first)
        try:
            cursor.execute("DELETE FROM trend_labels WHERE sku = ?", (sku,))
            cursor.execute(
                """
                INSERT INTO trend_labels (sku, trend_status, confidence_score, model_used, growth_rate)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sku, result["trend_status"], result["confidence_score"], result["model_used"], result["growth_rate_recent"])
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

        product_info = dict(prod_row) if prod_row else {}

        return {
            "sku": sku,
            "product_info": product_info,
            "trend_status": result["trend_status"],
            "confidence_score": result["confidence_score"],
            "probabilities": result["probabilities"],
            "growth_rate_recent": result["growth_rate_recent"],
            "latest_daily_mean_28d": result["latest_daily_mean_28d"],
            "model_used": result["model_used"],
            "feature_importance": result.get("feature_importance", [])
        }
