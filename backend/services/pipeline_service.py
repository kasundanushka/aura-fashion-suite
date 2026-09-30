"""
Pipeline Service for Flask Backend.
Handles file uploads, runs cleaning and feature engineering pipelines,
and refreshes database tables.
"""

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_pipeline.clean_data import run_cleaning_pipeline
from src.data_pipeline.feature_engineering import generate_feature_matrix
from database.init_db import populate_database_from_csv

logger = logging.getLogger("PipelineService")

class PipelineService:
    """Coordinates automated retraining and data ingestion runs."""

    # Canonical schema mapping patterns: canonical column name -> list of aliases & variations
    CANONICAL_SCHEMA_PATTERNS = {
        "date": [
            "date", "purchase_date", "purchasedate", "purchased_date", "purchased_at", "purchase_time",
            "order_date", "orderdate", "ordered_at", "order_time", "sale_date", "sales_date", "saledate",
            "transaction_date", "trans_date", "transactiondate", "tx_date", "invoice_date", "invoicedate",
            "bill_date", "billing_date", "timestamp", "time", "day", "datetime", "date_time", "created_at",
            "created_date", "event_date", "period", "posting_date", "entry_date", "sold_at", "sold_date"
        ],
        "sku": [
            "sku", "product_id", "productid", "product_code", "productcode", "item_id", "itemid",
            "item_code", "itemcode", "article_id", "articleid", "stock_code", "stockcode", "barcode",
            "upc", "ean", "asin", "isbn", "id", "part_number", "model", "style_number", "style", "item"
        ],
        "quantity_sold": [
            "quantity_sold", "quantitysold", "qty_sold", "qtysold", "units_sold", "unitssold",
            "sales_quantity", "sales_qty", "qty", "quantity", "units", "volume", "amount_sold",
            "count", "sales_count", "items_sold", "orders", "num_orders"
        ],
        "unit_price": [
            "unit_price", "unitprice", "price", "sale_price", "saleprice", "selling_price", "sellingprice",
            "current_price", "currentprice", "original_price", "retail_price", "retailprice", "rate",
            "item_price", "product_price", "cost_per_unit"
        ],
        "total_amount": [
            "total_amount", "totalamount", "total", "amount", "revenue", "sales_amount", "salesamount",
            "total_sales", "totalsales", "subtotal", "grand_total", "line_total", "total_spend"
        ],
        "discount_applied": [
            "discount_applied", "discountapplied", "discount", "discount_pct", "discount_percent",
            "discount_percentage", "discount_rate", "markdown_percentage", "markdown_percent",
            "markdown_pct", "markdown", "disc", "promo_discount", "rebate"
        ],
        "current_stock": [
            "current_stock", "currentstock", "stock", "stock_quantity", "stockquantity", "stock_level",
            "stocklevel", "inventory_level", "inventorylevel", "inventory", "on_hand", "quantity_on_hand",
            "qty_on_hand", "available_stock"
        ],
        "category": [
            "category", "item_category", "itemcategory", "product_category", "productcategory",
            "department", "dept", "group", "class", "type", "family", "product_type"
        ],
        "subcategory": [
            "subcategory", "sub_category", "sub_dept", "subgroup", "sub_type", "sub_class"
        ],
        "season": [
            "season", "seasonality", "collection"
        ],
        "name": [
            "name", "product_name", "productname", "item_name", "itemname", "title", "description",
            "label", "product_title"
        ],
        "brand": [
            "brand", "brand_name", "manufacturer", "make"
        ]
    }

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps common alternative column names to the internal schema expected
        by the cleaning pipeline (date, sku, quantity_sold, unit_price, etc.).
        Applies exact and fuzzy normalized matching (case/space/symbol insensitive)
        with automated datetime content inference fallback.
        """
        # Build normalized lookup map: cleaned_pattern -> canonical_target
        pattern_map = {}
        for target, aliases in PipelineService.CANONICAL_SCHEMA_PATTERNS.items():
            for alias in aliases:
                key = str(alias).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
                if key not in pattern_map:
                    pattern_map[key] = target

        rename_map = {}
        matched_targets = set()

        # Step 0: Register targets that already exist directly in df.columns
        for col in df.columns:
            c_clean = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            if c_clean in PipelineService.CANONICAL_SCHEMA_PATTERNS:
                matched_targets.add(c_clean)
                if col != c_clean:
                    rename_map[col] = c_clean

        # Step 1: Direct or normalized pattern matching for missing targets
        for col in df.columns:
            if col in rename_map:
                continue
            clean_key = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            if clean_key in pattern_map:
                target = pattern_map[clean_key]
                # Only rename if target column is not yet satisfied
                if target not in matched_targets:
                    rename_map[col] = target
                    matched_targets.add(target)

        # Step 2: Fallback for 'date' if still not identified
        if "date" not in matched_targets and "date" not in df.columns and "date" not in rename_map.values():
            for col in df.columns:
                if col in rename_map:
                    continue
                c_norm = str(col).strip().lower().replace(" ", "_").replace("-", "_")
                if any(substr in c_norm for substr in ["date", "time", "timestamp", "day"]) and not any(ex in c_norm for ex in ["lead_time", "delivery_time", "duration", "times"]):
                    try:
                        sample = df[col].dropna().head(10)
                        if len(sample) > 0:
                            pd.to_datetime(sample, errors="raise")
                            rename_map[col] = "date"
                            matched_targets.add("date")
                            logger.info(f"Auto-inferred 'date' column from datetime values in '{col}'")
                            break
                    except Exception:
                        pass

        # Step 3: Fallback for 'unit_price' if still not identified
        if "unit_price" not in matched_targets and "unit_price" not in df.columns and "unit_price" not in rename_map.values():
            for col in df.columns:
                if col in rename_map:
                    continue
                c_norm = str(col).strip().lower().replace(" ", "_").replace("-", "_")
                if any(substr in c_norm for substr in ["price", "cost", "rate", "retail"]):
                    rename_map[col] = "unit_price"
                    matched_targets.add("unit_price")
                    logger.info(f"Auto-inferred 'unit_price' column from '{col}'")
                    break

        if rename_map:
            logger.info(f"Auto-mapped columns: {rename_map}")
            df = df.rename(columns=rename_map)

        return df

    @staticmethod
    def _sync_products_and_stock(df: pd.DataFrame, raw_dir: str):
        """
        Synchronizes products.csv and stock_levels.csv with the SKUs present
        in the uploaded sales dataset so that the catalog and stock levels
        always match the forecastable items.
        """
        prod_path = os.path.join(raw_dir, "products.csv")
        stock_path = os.path.join(raw_dir, "stock_levels.csv")

        existing_prods = {}
        if os.path.exists(prod_path):
            try:
                ep_df = pd.read_csv(prod_path)
                for _, r in ep_df.iterrows():
                    existing_prods[str(r["sku"])] = dict(r)
            except Exception:
                pass

        existing_stocks = {}
        if os.path.exists(stock_path):
            try:
                es_df = pd.read_csv(stock_path)
                for _, r in es_df.iterrows():
                    existing_stocks[str(r["sku"])] = dict(r)
            except Exception:
                pass

        now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Group by SKU in one pass instead of repetitive filtering
        df_clean = df.copy()
        df_clean["sku"] = df_clean["sku"].astype(str).str.strip()
        
        agg_dict = {}
        if "unit_price" in df_clean.columns:
            agg_dict["unit_price"] = "mean"
        if "category" in df_clean.columns:
            agg_dict["category"] = "first"
        if "subcategory" in df_clean.columns:
            agg_dict["subcategory"] = "first"
        if "season" in df_clean.columns:
            agg_dict["season"] = "first"
        if "name" in df_clean.columns:
            agg_dict["name"] = "first"
        if "date" in df_clean.columns:
            agg_dict["date"] = "min"
        if "current_stock" in df_clean.columns:
            agg_dict["current_stock"] = "last"

        if agg_dict:
            grouped = df_clean.groupby("sku", as_index=False).agg(agg_dict)
        else:
            grouped = pd.DataFrame({"sku": df_clean["sku"].unique()})

        synced_products = []
        synced_stocks = []

        for _, row in grouped.iterrows():
            s = str(row["sku"])
            
            # Product metadata
            if s in existing_prods:
                p_item = existing_prods[s].copy()
                if "unit_price" in row and pd.notna(row["unit_price"]):
                    avg_p = float(row["unit_price"])
                    p_item["retail_price"] = round(avg_p, 2)
                    p_item["cost_price"] = round(avg_p * 0.60, 2)
            else:
                cat = str(row["category"]) if "category" in row and pd.notna(row["category"]) else "General"
                subcat = str(row["subcategory"]) if "subcategory" in row and pd.notna(row["subcategory"]) else cat
                season = str(row["season"]) if "season" in row and pd.notna(row["season"]) else "All-Season"
                name = str(row["name"]) if "name" in row and pd.notna(row["name"]) else f"{cat} Item ({s})"
                avg_p = float(row["unit_price"]) if "unit_price" in row and pd.notna(row["unit_price"]) else 50.0
                retail_p = round(avg_p, 2)
                cost_p = round(retail_p * 0.60, 2)
                launch_d = str(row["date"])[:10] if "date" in row and pd.notna(row["date"]) else "2024-01-01"

                p_item = {
                    "sku": s,
                    "name": name,
                    "category": cat,
                    "subcategory": subcat,
                    "season": season,
                    "material": "Standard Quality",
                    "cost_price": cost_p,
                    "retail_price": retail_p,
                    "launch_date": launch_d
                }
            synced_products.append(p_item)

            # Stock levels
            if "current_stock" in row and pd.notna(row["current_stock"]):
                stock_val = int(row["current_stock"])
            elif s in existing_stocks and pd.notna(existing_stocks[s].get("current_stock")):
                stock_val = int(existing_stocks[s]["current_stock"])
            else:
                stock_val = 100

            synced_stocks.append({
                "sku": s,
                "current_stock": stock_val,
                "last_updated": now_str
            })

        pd.DataFrame(synced_products).to_csv(prod_path, index=False)
        pd.DataFrame(synced_stocks).to_csv(stock_path, index=False)
        logger.info(f"Synchronized {len(synced_products)} SKUs to {prod_path} and {stock_path}")

    @staticmethod
    def process_sales_upload(file_storage, target_filename="sales_data.csv", uploaded_by="System Admin") -> dict:
        """
        Validates uploaded file, normalizes column names, synchronizes product
        catalogs & stock levels, saves to raw storage, archives a copy, records upload history,
        and re-executes the data pipeline.
        """
        import time
        from werkzeug.utils import secure_filename
        
        raw_dir = "data/raw"
        uploads_dir = "data/uploads"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        dest_path = os.path.join(raw_dir, target_filename)

        original_name = getattr(file_storage, "filename", None) or target_filename
        clean_name = secure_filename(original_name) or "sales_data.csv"
        archive_name = f"{int(time.time())}_{clean_name}"
        archive_path = os.path.join(uploads_dir, archive_name)

        try:
            # Read the CSV into a DataFrame first for column normalization
            df = pd.read_csv(file_storage)
            logger.info(f"Uploaded CSV columns: {list(df.columns)}")

            # Drop index artifact columns like 'Unnamed: 0'
            unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            # Auto-map alternative column names to the expected schema
            df = PipelineService._normalize_columns(df)

            # Smart fallback if 'sku' is missing:
            if "sku" not in df.columns:
                sku_source_candidates = [
                    "item_category", "ItemCategory", "category", "Category",
                    "product", "Product", "item", "Item", "name", "Name",
                    "product_name", "Product Name", "description", "Description",
                    "title", "Title", "department", "Department"
                ]
                for cand in sku_source_candidates:
                    match = next((c for c in df.columns if c.lower() == cand.lower()), None)
                    if match:
                        df["sku"] = df[match].astype(str).str.strip()
                        logger.info(f"Auto-derived 'sku' from '{match}' column.")
                        if "category" not in df.columns:
                            df["category"] = df["sku"]
                        if "name" not in df.columns:
                            df["name"] = df["sku"]
                        break

                # Fallback to any non-id string column
                if "sku" not in df.columns:
                    excluded = {
                        "date", "invoiceno", "invoice_no", "invoice", "paymentmethod",
                        "payment_method", "store", "total_amount", "totalamount", "amount"
                    }
                    potential = [c for c in df.columns if c.lower() not in excluded and df[c].dtype == object]
                    if potential:
                        chosen = potential[0]
                        df["sku"] = df[chosen].astype(str).str.strip()
                        logger.info(f"Auto-derived 'sku' from fallback column '{chosen}'.")
                        if "category" not in df.columns:
                            df["category"] = df["sku"]
                        if "name" not in df.columns:
                            df["name"] = df["sku"]

            # Smart fallback if 'quantity_sold' is missing (default 1 unit per invoice transaction)
            if "quantity_sold" not in df.columns:
                df["quantity_sold"] = 1
                logger.info("Defaulted 'quantity_sold' to 1 for invoice line items.")

            # Smart fallback if 'unit_price' is missing but 'total_amount' is present
            if "unit_price" not in df.columns:
                if "total_amount" in df.columns:
                    qty = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(1).replace(0, 1)
                    amt = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)
                    df["unit_price"] = (amt / qty).round(2)
                    logger.info("Auto-calculated 'unit_price' from total_amount / quantity_sold.")
                else:
                    df["unit_price"] = 10.0
                    logger.info("Defaulted 'unit_price' to 10.0.")

            # Smart fallback and scale normalization for discount
            if "discount_applied" in df.columns:
                disc_num = pd.to_numeric(df["discount_applied"], errors="coerce").fillna(0.0)
                if disc_num.max() > 1.0 and disc_num.max() <= 100.0:
                    df["discount_applied"] = (disc_num / 100.0).round(4)
                    logger.info("Normalized discount percentage scale (0-100% -> 0.0-1.0)")
                else:
                    df["discount_applied"] = disc_num.clip(lower=0.0, upper=1.0).round(4)
            else:
                df["discount_applied"] = 0.0

            # Verify all required columns are present after mapping & fallback
            required = {"date", "sku", "quantity_sold", "unit_price"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"Sales dataset is missing required columns even after auto-mapping: {missing}. "
                    f"Uploaded columns were: {list(df.columns)}. "
                    f"Expected: date, sku, quantity_sold, unit_price"
                )

            # Save the normalized CSV so the downstream pipeline reads clean headers
            df.to_csv(dest_path, index=False)
            # Save archived backup copy
            df.to_csv(archive_path, index=False)
            logger.info(f"Saved normalized sales CSV to {dest_path} and archive {archive_path} ({len(df)} rows)")

            # Sync products and stock levels for all SKUs in the uploaded file
            PipelineService._sync_products_and_stock(df, raw_dir)

            # Run pipeline stages
            run_cleaning_pipeline(raw_dir="data/raw", processed_dir="data/processed")
            generate_feature_matrix(processed_dir="data/processed")
            populate_database_from_csv()

            file_size = os.path.getsize(dest_path)
            row_count = len(df)
            detected_cols = ", ".join(list(df.columns))

            # Record in CSV upload history
            try:
                from database.init_db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO csv_upload_history (
                        filename, saved_path, uploaded_by, file_size_bytes, row_count,
                        columns_detected, status, pipeline_status, uploaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    original_name,
                    archive_path,
                    uploaded_by,
                    file_size,
                    row_count,
                    detected_cols,
                    "SUCCESS",
                    "Ingested & Models Retrained"
                ))
                conn.commit()
                conn.close()
            except Exception as dberr:
                logger.warning(f"Could not record csv_upload_history: {dberr}")

            # Invalidate forecast and trend model memory caches
            try:
                from backend.services.forecast_service import clear_forecast_cache
                clear_forecast_cache()
            except Exception:
                pass

            try:
                from src.classification.trend_classifier import reset_trend_classifier
                reset_trend_classifier()
            except Exception:
                pass

            # Pre-compute and sync reorder recommendations immediately
            try:
                from backend.services.inventory_service import InventoryService
                InventoryService.get_all_reorder_alerts()
            except Exception as ie:
                logger.warning(f"Could not refresh inventory recommendations: {ie}")

            return {
                "status": "success",
                "message": "Sales data successfully ingested and pipeline refreshed.",
                "file_size_bytes": file_size,
                "row_count": row_count,
                "filename": original_name
            }

        except Exception as err:
            # Log failed upload attempt
            try:
                from database.init_db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO csv_upload_history (
                        filename, saved_path, uploaded_by, file_size_bytes, row_count,
                        columns_detected, status, pipeline_status, error_message, uploaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    original_name,
                    None,
                    uploaded_by,
                    0,
                    0,
                    None,
                    "FAILED",
                    "Validation / Pipeline Error",
                    str(err)
                ))
                conn.commit()
                conn.close()
            except Exception:
                pass
            raise err

    @staticmethod
    def trigger_full_pipeline_refresh() -> dict:
        """
        Executes full data cleaning, feature engineering, and database refresh
        from currently stored raw datasets. Clears caches and synchronizes alerts.
        """
        run_cleaning_pipeline(raw_dir="data/raw", processed_dir="data/processed")
        generate_feature_matrix(processed_dir="data/processed")
        populate_database_from_csv()

        try:
            from backend.services.forecast_service import clear_forecast_cache
            clear_forecast_cache()
        except Exception:
            pass

        try:
            from src.classification.trend_classifier import reset_trend_classifier
            reset_trend_classifier()
        except Exception:
            pass

        try:
            from backend.services.inventory_service import InventoryService
            alerts = InventoryService.get_all_reorder_alerts()
            urgent = sum(1 for a in alerts if a.get("alert_level") == "URGENT_REORDER")
        except Exception:
            urgent = 0

        return {
            "status": "success",
            "message": "AI Pipeline fully retrained, feature matrix regenerated, and inventory alerts synchronized.",
            "urgent_alerts_count": urgent
        }
