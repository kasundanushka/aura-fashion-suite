-- SQLite Database Schema for Clothing AI Inventory Optimization System

-- 1. Products Table
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    season TEXT NOT NULL,
    material TEXT NOT NULL,
    cost_price REAL NOT NULL,
    retail_price REAL NOT NULL,
    launch_date TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Sales History Table
CREATE TABLE IF NOT EXISTS sales_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    sku TEXT NOT NULL,
    quantity_sold INTEGER NOT NULL CHECK (quantity_sold >= 0),
    unit_price REAL NOT NULL CHECK (unit_price >= 0),
    revenue REAL GENERATED ALWAYS AS (quantity_sold * unit_price) STORED,
    discount_applied REAL DEFAULT 0.0,
    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sales_date_sku ON sales_history(date, sku);
CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales_history(sku);

-- 3. Stock Levels Table
CREATE TABLE IF NOT EXISTS stock_levels (
    sku TEXT PRIMARY KEY,
    current_stock INTEGER NOT NULL CHECK (current_stock >= 0),
    safety_stock INTEGER DEFAULT 0,
    reorder_point INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE CASCADE
);

-- 4. Forecasts Table
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    forecast_date TEXT NOT NULL,
    predicted_quantity REAL NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    model_used TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_forecast_sku_date ON forecasts(sku, forecast_date);

-- 5. Trend Labels Table
CREATE TABLE IF NOT EXISTS trend_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    trend_status TEXT NOT NULL CHECK (trend_status IN ('rising', 'stable', 'declining')),
    confidence_score REAL NOT NULL,
    model_used TEXT NOT NULL,
    growth_rate REAL,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trend_sku ON trend_labels(sku);

-- 6. Reorder Recommendations Table
CREATE TABLE IF NOT EXISTS reorder_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    recommended_quantity INTEGER NOT NULL CHECK (recommended_quantity >= 0),
    alert_level TEXT NOT NULL CHECK (alert_level IN ('URGENT_REORDER', 'REORDER_SOON', 'SUFFICIENT_STOCK', 'OVERSTOCKED')),
    forecasted_demand REAL NOT NULL,
    current_stock INTEGER NOT NULL,
    safety_stock INTEGER NOT NULL,
    trend_factor REAL NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sku) REFERENCES products (sku) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reorder_sku ON reorder_recommendations(sku);

-- 7. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')) DEFAULT 'user',
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 8. Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT,
    action TEXT NOT NULL,
    details TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- 9. CSV Upload History Table
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
);

CREATE INDEX IF NOT EXISTS idx_upload_history_date ON csv_upload_history(uploaded_at);

-- 10. Buyer Measurements Table
CREATE TABLE IF NOT EXISTS buyer_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('male', 'female')),
    height REAL,
    chest REAL,
    waist REAL,
    hips REAL,
    shoulder_width REAL,
    arm_length REAL,
    leg_length REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_buyer_measurements_user ON buyer_measurements(user_id);
