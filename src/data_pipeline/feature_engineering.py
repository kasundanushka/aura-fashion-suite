"""
Feature Engineering Module for Clothing AI System.
Transforms cleaned time series data into rich feature matrices for:
1. Time-series demand forecasting (Prophet & Baseline Regression)
2. Supervised trend classification (Random Forest & KNN)
3. Inventory reorder optimization calculations
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling window statistics (7d, 14d, 28d) per SKU with optimized performance."""
    df = df.sort_values(by=["sku", "date"]).reset_index(drop=True).copy()
    
    # Check if data is mostly single-transaction per SKU (sparse catalog)
    sku_counts = df["sku"].value_counts()
    is_sparse = (len(sku_counts) > 50 and sku_counts.max() <= 3)

    if is_sparse:
        qty = df["quantity_sold"].astype(float)
        for window in [7, 14, 28]:
            df[f"rolling_mean_{window}d"] = qty
            df[f"rolling_std_{window}d"] = 1.0
            df[f"rolling_max_{window}d"] = qty
        for lag in [1, 7, 14, 28]:
            df[f"lag_{lag}d"] = qty
        return df

    # Native grouped rolling (no Python lambda overhead)
    grouped = df.groupby("sku")["quantity_sold"]
    shifted = grouped.shift(1)
    
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}d"] = (
            shifted.groupby(df["sku"]).rolling(window, min_periods=1).mean()
        ).reset_index(drop=True).fillna(df["quantity_sold"].astype(float))
        df[f"rolling_std_{window}d"] = (
            shifted.groupby(df["sku"]).rolling(window, min_periods=1).std()
        ).reset_index(drop=True).fillna(0.0)
        df[f"rolling_max_{window}d"] = (
            shifted.groupby(df["sku"]).rolling(window, min_periods=1).max()
        ).reset_index(drop=True).fillna(df["quantity_sold"].astype(float))

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}d"] = grouped.shift(lag).bfill().fillna(df["quantity_sold"].astype(float))

    return df

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical temporal signals."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Cyclical Sine/Cosine transformations for smooth periodicity
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    return df

def compute_product_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates days since product launch and pricing elasticity features."""
    df = df.copy()
    if "launch_date" in df.columns:
        launch_dt = pd.to_datetime(df["launch_date"], errors="coerce")
        df["days_since_launch"] = (df["date"] - launch_dt).dt.days.clip(lower=0).fillna(0)
    else:
        df["days_since_launch"] = 0

    # Price ratio relative to retail price
    if "retail_price" in df.columns and "unit_price" in df.columns:
        retail = pd.to_numeric(df["retail_price"], errors="coerce").fillna(df["unit_price"])
        retail = retail.replace(0, 1.0)
        df["price_ratio"] = (df["unit_price"] / retail).clip(0.0, 2.0).fillna(1.0)
    else:
        df["price_ratio"] = 1.0

    return df

def compute_category_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily category-level sales momentum to capture broader fashion trends."""
    df = df.copy()
    if "category" in df.columns and df["category"].notna().any():
        cat_daily = df.groupby(["category", "date"])["quantity_sold"].mean().reset_index()
        cat_daily.rename(columns={"quantity_sold": "category_daily_avg_demand"}, inplace=True)
        # Shift category momentum by 1 day to prevent data leakage
        cat_daily["category_daily_avg_demand"] = (
            cat_daily.groupby("category")["category_daily_avg_demand"].shift(1).fillna(0.0)
        )
        df = df.merge(cat_daily, on=["category", "date"], how="left")
    
    if "category_daily_avg_demand" not in df.columns:
        df["category_daily_avg_demand"] = 0.0
    else:
        df["category_daily_avg_demand"] = df["category_daily_avg_demand"].fillna(0.0)

    return df

def assign_trend_classification_labels(df: pd.DataFrame, lookback_days: int = 56) -> pd.DataFrame:
    """
    Assigns ground-truth trend label ('rising', 'stable', 'declining') based on
    the slope/rate of sales growth over a lookback window (default 8 weeks = 56 days).
    """
    df = df.sort_values(by=["sku", "date"]).reset_index(drop=True).copy()
    
    sku_counts = df["sku"].value_counts()
    is_sparse = (len(sku_counts) > 50 and sku_counts.max() <= 3)

    if is_sparse:
        df["sales_growth_rate"] = 0.0
        df["trend_label"] = "stable"
        return df

    grouped = df.groupby("sku")["quantity_sold"]
    recent_28d = (
        grouped.shift(1).groupby(df["sku"]).rolling(28, min_periods=14).mean()
    ).reset_index(drop=True)
    prior_28d = (
        grouped.shift(29).groupby(df["sku"]).rolling(28, min_periods=14).mean()
    ).reset_index(drop=True)

    growth_rate = (recent_28d - prior_28d) / prior_28d.clip(lower=1.0)
    df["sales_growth_rate"] = growth_rate.fillna(0.0)

    # Classify labels
    conditions = [
        df["sales_growth_rate"] > 0.10,
        df["sales_growth_rate"] < -0.10
    ]
    choices = ["rising", "declining"]
    df["trend_label"] = np.select(conditions, choices, default="stable")

    return df

def generate_feature_matrix(processed_dir="data/processed"):
    """Loads clean sales, executes feature engineering pipeline, and persists feature matrix."""
    clean_sales_file = os.path.join(processed_dir, "clean_sales.csv")
    if not os.path.exists(clean_sales_file):
        raise FileNotFoundError(f"Clean sales file not found at {clean_sales_file}. Run clean_data first.")

    df = pd.read_csv(clean_sales_file)
    logger.info(f"Loaded {len(df)} cleaned records for feature engineering.")

    df = compute_temporal_features(df)
    df = compute_product_lifecycle_features(df)
    df = compute_rolling_features(df)
    df = compute_category_aggregates(df)
    df = assign_trend_classification_labels(df)

    out_path = os.path.join(processed_dir, "feature_matrix.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"[OK] Feature matrix created with {df.shape[1]} columns and {len(df)} rows.")
    logger.info(f"     Saved to: {out_path}")

    # Summary of trend distribution
    latest_snapshot = df.sort_values("date").groupby("sku").last()
    logger.info(f"Latest SKU Trend Distribution:\n{latest_snapshot['trend_label'].value_counts()}")

    return df

if __name__ == "__main__":
    generate_feature_matrix()
