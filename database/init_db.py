"""
Database Initialization & Population Script.
Executes schema.sql to create tables and populates products, sales history,
and initial stock levels into SQLite clothing.db.
"""

import os
import sys
import sqlite3
import pandas as pd
from werkzeug.security import generate_password_hash

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_PATH = os.path.join(os.path.dirname(__file__), "clothing.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def get_db_connection(db_path=DB_PATH):
    """Creates and returns a SQLite database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def seed_default_users(conn):
    """Seeds default admin and user accounts if they do not exist."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@aura.ai'")
    if cursor.fetchone()[0] == 0:
        admin_pwd_hash = generate_password_hash("admin123")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, avatar_url)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "System Administrator",
            "admin@aura.ai",
            admin_pwd_hash,
            "admin",
            "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop"
        ))
        print("[OK] Seeded default admin user: admin@aura.ai")

    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'demo@aura.ai'")
    if cursor.fetchone()[0] == 0:
        demo_pwd_hash = generate_password_hash("demo123")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, avatar_url)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "Demo Fashion Buyer",
            "demo@aura.ai",
            demo_pwd_hash,
            "user",
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop"
        ))
        print("[OK] Seeded default demo user: demo@aura.ai")
    conn.commit()

def migrate_users_table(conn):
    """Ensures users table has role and avatar_url columns."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    cols = [col[1] for col in cursor.fetchall()]
    if cols and "role" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        print("[OK] Added 'role' column to users table.")
    if cols and "avatar_url" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
        print("[OK] Added 'avatar_url' column to users table.")
    conn.commit()

def init_upload_history_table(conn):
    """Ensures csv_upload_history table exists and has seed data if empty."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS csv_upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            saved_path TEXT,
            uploaded_by TEXT DEFAULT 'System Admin',
            file_size_bytes INTEGER DEFAULT 0,
            row_count INTEGER DEFAULT 0,
            columns_detected TEXT,
            status TEXT DEFAULT 'SUCCESS',
            pipeline_status TEXT DEFAULT 'Ingested & Retrained',
            error_message TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_history_date ON csv_upload_history(uploaded_at)")

    cursor.execute("SELECT COUNT(*) FROM csv_upload_history")
    if cursor.fetchone()[0] == 0:
        sales_path = "data/raw/sales_data.csv"
        size = os.path.getsize(sales_path) if os.path.exists(sales_path) else 6191465
        rows = 14620
        cols_str = "date, sku, quantity_sold, unit_price, total_amount, discount_applied"
        if os.path.exists(sales_path):
            try:
                df_temp = pd.read_csv(sales_path, nrows=5)
                cols_str = ", ".join(df_temp.columns)
            except Exception:
                pass

        cursor.execute("""
            INSERT INTO csv_upload_history (
                filename, saved_path, uploaded_by, file_size_bytes, row_count,
                columns_detected, status, pipeline_status, uploaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '2026-09-05 08:00:00')
        """, (
            "sales_data.csv",
            "data/raw/sales_data.csv",
            "admin@aura.ai",
            size,
            rows,
            cols_str,
            "SUCCESS",
            "Initial Dataset Ingested & Trained"
        ))
    conn.commit()

def init_database(db_path=DB_PATH, schema_path=SCHEMA_PATH):
    """Executes schema.sql to initialize database tables and seed users."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    migrate_users_table(conn)
    seed_default_users(conn)
    init_upload_history_table(conn)
    conn.close()
    print(f"[OK] Initialized database schema at {db_path}")

def sync_reorder_recommendations(conn, alerts: list):
    """
    Persists computed reorder alerts into reorder_recommendations table
    so that Admin and Dashboard KPIs always reflect active threshold breaches.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reorder_recommendations")
    for a in alerts:
        cursor.execute("""
            INSERT INTO reorder_recommendations (
                sku, recommended_quantity, alert_level, forecasted_demand,
                current_stock, safety_stock, trend_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(a.get("sku")),
            int(a.get("recommended_quantity", 0)),
            str(a.get("alert_level", "SUFFICIENT_STOCK")),
            float(a.get("forecasted_demand", 0.0)),
            int(a.get("current_stock", 0)),
            int(a.get("safety_stock", 0)),
            float(a.get("trend_factor", 1.0))
        ))
    conn.commit()
    print(f"[OK] Synced {len(alerts)} records into 'reorder_recommendations' table.")

def populate_database_from_csv(
    db_path=DB_PATH,
    products_csv=None,
    sales_csv=None,
    stock_csv=None
):
    """Populates products, sales_history, and stock_levels from CSVs preserving schema constraints."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Prefer processed clean products and sales if available
    if products_csv is None:
        if os.path.exists("data/processed/clean_products.csv"):
            products_csv = "data/processed/clean_products.csv"
        else:
            products_csv = "data/raw/products.csv"

    if sales_csv is None:
        if os.path.exists("data/processed/clean_sales.csv"):
            sales_csv = "data/processed/clean_sales.csv"
        else:
            sales_csv = "data/raw/sales_data.csv"

    if stock_csv is None:
        stock_csv = "data/raw/stock_levels.csv"

    if products_csv and os.path.exists(products_csv):
        df_prod = pd.read_csv(products_csv)
        cursor.execute("DELETE FROM products")
        df_prod.to_sql("products", conn, if_exists="append", index=False)
        print(f"[OK] Populated {len(df_prod)} rows into 'products' table from {products_csv}.")

    if sales_csv and os.path.exists(sales_csv):
        df_sales = pd.read_csv(sales_csv)
        cursor.execute("DELETE FROM sales_history")
        df_sales.to_sql("sales_history", conn, if_exists="append", index=False)
        print(f"[OK] Populated {len(df_sales)} rows into 'sales_history' table from {sales_csv}.")

    if stock_csv and os.path.exists(stock_csv):
        df_stock = pd.read_csv(stock_csv)
        cursor.execute("DELETE FROM stock_levels")
        df_stock.to_sql("stock_levels", conn, if_exists="append", index=False)
        print(f"[OK] Populated {len(df_stock)} rows into 'stock_levels' table from {stock_csv}.")

    # Ensure all products have an entry in stock_levels table
    cursor.execute("""
        INSERT OR IGNORE INTO stock_levels (sku, current_stock, last_updated)
        SELECT p.sku, 100, datetime('now')
        FROM products p
        LEFT JOIN stock_levels s ON p.sku = s.sku
        WHERE s.sku IS NULL
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    populate_database_from_csv()
