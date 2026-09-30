"""
Data Cleaning and Validation Module for Clothing AI System.
Validates input data schemas, handles missing values, filters corrupt rows,
and aligns continuous time-series index per SKU.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataCleaning")

REQUIRED_SALES_COLUMNS = {"date", "sku", "quantity_sold", "unit_price"}
REQUIRED_PRODUCT_COLUMNS = {"sku", "name", "category", "subcategory", "season", "cost_price", "retail_price"}

def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates sales dataframe schema and data constraints:
      - Checks required columns.
      - Converts 'date' to datetime and drops invalid dates.
      - Rejects negative quantities or negative unit prices.
      - Fills missing discount values with 0.0.
    """
    if df is None or df.empty:
        raise ValueError("Sales dataset is empty or None.")

    missing_cols = REQUIRED_SALES_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Sales dataset is missing required columns: {missing_cols}")

    initial_len = len(df)
    df_clean = df.copy()

    # 1. Parse and validate date
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    invalid_dates = df_clean["date"].isna()
    if invalid_dates.sum() > 0:
        logger.warning(f"Dropping {invalid_dates.sum()} rows with unparseable dates.")
        df_clean = df_clean[~invalid_dates]

    # 2. Filter invalid numbers
    df_clean["quantity_sold"] = pd.to_numeric(df_clean["quantity_sold"], errors="coerce")
    df_clean["unit_price"] = pd.to_numeric(df_clean["unit_price"], errors="coerce")

    valid_mask = (
        (df_clean["quantity_sold"] >= 0) &
        (df_clean["unit_price"] >= 0) &
        df_clean["sku"].notna() &
        (df_clean["sku"].str.strip() != "")
    )
    dropped_count = initial_len - valid_mask.sum()
    if dropped_count > 0:
        logger.warning(f"Filtered out {dropped_count} invalid or corrupt sales records.")

    df_clean = df_clean[valid_mask].copy()
    df_clean["quantity_sold"] = df_clean["quantity_sold"].astype(int)
    df_clean["unit_price"] = df_clean["unit_price"].astype(float)
    df_clean["sku"] = df_clean["sku"].astype(str).str.strip()

    if "discount_applied" in df_clean.columns:
        df_clean["discount_applied"] = pd.to_numeric(df_clean["discount_applied"], errors="coerce").fillna(0.0)
    else:
        df_clean["discount_applied"] = 0.0

    return df_clean

def validate_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates product catalog schema and constraints."""
    if df is None or df.empty:
        raise ValueError("Product dataset is empty or None.")

    missing_cols = REQUIRED_PRODUCT_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Product dataset is missing required columns: {missing_cols}")

    df_clean = df.copy()
    df_clean["sku"] = df_clean["sku"].astype(str).str.strip()
    df_clean["cost_price"] = pd.to_numeric(df_clean["cost_price"], errors="coerce").fillna(0.0)
    df_clean["retail_price"] = pd.to_numeric(df_clean["retail_price"], errors="coerce").fillna(0.0)

    # Remove duplicates on primary key 'sku'
    df_clean = df_clean.drop_duplicates(subset=["sku"], keep="last")
    return df_clean

def fill_missing_dates_per_sku(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures continuous daily records for each SKU where appropriate.
    Consolidates multiple records on the same date (e.g., multi-store or multi-transaction),
    and fills timeline gaps between a SKU's first and last observed sale.
    Avoids phantom row explosion for sparse/one-off SKU transactions.
    """
    if df_sales is None or df_sales.empty:
        return df_sales

    # 1. Consolidate multiple records per sku and date
    df_daily = df_sales.groupby(["sku", "date"], as_index=False).agg({
        "quantity_sold": "sum",
        "unit_price": "mean",
        "discount_applied": "mean"
    })
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    all_skus = df_daily["sku"].unique()
    n_skus = len(all_skus)

    reindexed_frames = []
    grouped = df_daily.groupby("sku")

    # For small SKU counts (like standard 20 demo SKUs), align across common dataset span
    if n_skus <= 50:
        common_min = df_daily["date"].min()
        common_max = df_daily["date"].max()
        full_date_idx = pd.date_range(start=common_min, end=common_max, freq="D")
        for sku, sub_df in grouped:
            sub = sub_df.set_index("date").reindex(full_date_idx)
            sub["sku"] = sku
            sub["quantity_sold"] = sub["quantity_sold"].fillna(0).astype(int)
            sub["unit_price"] = sub["unit_price"].ffill().bfill().fillna(25.0)
            sub["discount_applied"] = sub["discount_applied"].fillna(0.0)
            reindexed_frames.append(sub.reset_index().rename(columns={"index": "date"}))
    else:
        # Large/sparse catalog: only fill gaps for SKUs that actually have a multi-day span
        for sku, sub_df in grouped:
            s_min = sub_df["date"].min()
            s_max = sub_df["date"].max()
            span_days = (s_max - s_min).days
            if span_days >= 3 and len(sub_df) >= 2:
                sku_date_idx = pd.date_range(start=s_min, end=s_max, freq="D")
                sub = sub_df.set_index("date").reindex(sku_date_idx)
                sub["sku"] = sku
                sub["quantity_sold"] = sub["quantity_sold"].fillna(0).astype(int)
                sub["unit_price"] = sub["unit_price"].ffill().bfill().fillna(25.0)
                sub["discount_applied"] = sub["discount_applied"].fillna(0.0)
                reindexed_frames.append(sub.reset_index().rename(columns={"index": "date"}))
            else:
                # Sparse / single-transaction SKU: keep existing rows without inflating 365 zeros
                reindexed_frames.append(sub_df)

    aligned_df = pd.concat(reindexed_frames, ignore_index=True)
    logger.info(f"Aligned daily time series across {n_skus} SKUs. Total records: {len(aligned_df)}")
    return aligned_df

def run_cleaning_pipeline(raw_dir="data/raw", processed_dir="data/processed"):
    """Reads raw CSVs, validates, cleans, fills time gaps, and saves cleaned CSVs."""
    os.makedirs(processed_dir, exist_ok=True)
    
    sales_file = os.path.join(raw_dir, "sales_data.csv")
    products_file = os.path.join(raw_dir, "products.csv")

    if not os.path.exists(sales_file):
        raise FileNotFoundError(f"Sales data file not found at {sales_file}. Run generator or upload first.")

    raw_sales = pd.read_csv(sales_file)
    clean_sales = validate_sales_data(raw_sales)

    # Load or initialize products
    if os.path.exists(products_file):
        raw_products = pd.read_csv(products_file)
        clean_products = validate_product_data(raw_products)
    else:
        clean_products = pd.DataFrame(columns=list(REQUIRED_PRODUCT_COLUMNS) + ["material", "launch_date"])

    # Auto-derive product metadata for any SKU present in sales but missing in products
    sales_skus = set(clean_sales["sku"].unique())
    prod_skus = set(clean_products["sku"].unique())
    missing_skus = sales_skus - prod_skus

    if missing_skus:
        logger.info(f"Deriving product catalog entries for {len(missing_skus)} new SKUs from sales data...")
        new_records = []
        for s in missing_skus:
            s_rows = clean_sales[clean_sales["sku"] == s]
            cat = str(s_rows["category"].iloc[0]) if "category" in s_rows.columns and pd.notna(s_rows["category"].iloc[0]) else "General"
            subcat = str(s_rows["subcategory"].iloc[0]) if "subcategory" in s_rows.columns and pd.notna(s_rows["subcategory"].iloc[0]) else cat
            season = str(s_rows["season"].iloc[0]) if "season" in s_rows.columns and pd.notna(s_rows["season"].iloc[0]) else "All-Season"
            name = str(s_rows["name"].iloc[0]) if "name" in s_rows.columns and pd.notna(s_rows["name"].iloc[0]) else f"{cat} Item ({s})"
            avg_p = float(s_rows["unit_price"].mean()) if len(s_rows) > 0 else 50.0
            retail_p = round(avg_p, 2)
            cost_p = round(retail_p * 0.60, 2)
            launch_d = str(s_rows["date"].min())[:10] if len(s_rows) > 0 else "2022-01-01"

            new_records.append({
                "sku": s,
                "name": name,
                "category": cat,
                "subcategory": subcat,
                "season": season,
                "material": "Standard Quality",
                "cost_price": cost_p,
                "retail_price": retail_p,
                "launch_date": launch_d
            })
        clean_products = pd.concat([clean_products, pd.DataFrame(new_records)], ignore_index=True)

    # Filter clean_products to active sales SKUs if sales dataset was completely replaced
    if sales_skus:
        active_products = clean_products[clean_products["sku"].isin(sales_skus)].copy()
        if not active_products.empty:
            clean_products = active_products

    aligned_sales = fill_missing_dates_per_sku(clean_sales)

    # Merge product metadata into sales dataset
    merged_df = aligned_sales.merge(clean_products, on="sku", how="left")

    out_sales_path = os.path.join(processed_dir, "clean_sales.csv")
    out_prod_path = os.path.join(processed_dir, "clean_products.csv")

    merged_df.to_csv(out_sales_path, index=False)
    clean_products.to_csv(out_prod_path, index=False)
    clean_products.to_csv(products_file, index=False)

    logger.info(f"[OK] Cleaning pipeline complete. Processed sales saved to {out_sales_path}")
    return merged_df, clean_products

if __name__ == "__main__":
    run_cleaning_pipeline()
