"""
Synthetic Sales & Product Data Generator for Clothing AI System.
Generates realistic multi-category fashion sales history (18 months, 30 SKUs)
with distinct seasonalities, trend trajectories, promotional discounts, and noise.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_product_catalog():
    """
    Returns a predefined catalog of 30 realistic fashion SKUs across 5 categories.
    Each SKU defines its intrinsic seasonality, trend trajectory, price, and cost.
    """
    return [
        # --- Outerwear (Strong Winter / Autumn Seasonality) ---
        {
            "sku": "OUT-001", "name": "Classic Wool Overcoat", "category": "Outerwear",
            "subcategory": "Coats", "season": "Winter", "material": "Wool Blend",
            "cost_price": 65.0, "retail_price": 149.99, "launch_date": "2023-09-01",
            "base_demand": 12, "trend_type": "rising", "peak_months": [10, 11, 12, 1, 2],
            "initial_stock": 45
        },
        {
            "sku": "OUT-002", "name": "Waterproof Puffer Jacket", "category": "Outerwear",
            "subcategory": "Jackets", "season": "Winter", "material": "Recycled Polyester",
            "cost_price": 48.0, "retail_price": 119.99, "launch_date": "2023-08-15",
            "base_demand": 16, "trend_type": "stable", "peak_months": [11, 12, 1, 2],
            "initial_stock": 120
        },
        {
            "sku": "OUT-003", "name": "Vintage Leather Biker Jacket", "category": "Outerwear",
            "subcategory": "Jackets", "season": "All-Season", "material": "Faux Leather",
            "cost_price": 42.0, "retail_price": 99.99, "launch_date": "2023-01-10",
            "base_demand": 8, "trend_type": "declining", "peak_months": [3, 4, 9, 10],
            "initial_stock": 210
        },
        {
            "sku": "OUT-004", "name": "Double-Breasted Trench Coat", "category": "Outerwear",
            "subcategory": "Coats", "season": "Spring/Autumn", "material": "Cotton Gabardine",
            "cost_price": 55.0, "retail_price": 129.99, "launch_date": "2023-02-01",
            "base_demand": 10, "trend_type": "rising", "peak_months": [3, 4, 9, 10],
            "initial_stock": 25
        },
        {
            "sku": "OUT-005", "name": "Cropped Denim Trucker Jacket", "category": "Outerwear",
            "subcategory": "Jackets", "season": "All-Season", "material": "Cotton Denim",
            "cost_price": 28.0, "retail_price": 69.99, "launch_date": "2023-03-15",
            "base_demand": 14, "trend_type": "stable", "peak_months": [4, 5, 8, 9],
            "initial_stock": 95
        },
        {
            "sku": "OUT-006", "name": "Quilted Bomber Jacket", "category": "Outerwear",
            "subcategory": "Jackets", "season": "Spring/Autumn", "material": "Nylon",
            "cost_price": 34.0, "retail_price": 79.99, "launch_date": "2023-04-01",
            "base_demand": 7, "trend_type": "declining", "peak_months": [9, 10, 11],
            "initial_stock": 180
        },

        # --- Knitwear (Autumn / Winter / Trans-seasonal) ---
        {
            "sku": "KNT-001", "name": "Luxury Cashmere Crewneck", "category": "Knitwear",
            "subcategory": "Sweaters", "season": "Winter", "material": "100% Cashmere",
            "cost_price": 52.0, "retail_price": 120.00, "launch_date": "2023-09-15",
            "base_demand": 14, "trend_type": "rising", "peak_months": [10, 11, 12, 1],
            "initial_stock": 30
        },
        {
            "sku": "KNT-002", "name": "Chunky Cable-Knit Turtleneck", "category": "Knitwear",
            "subcategory": "Sweaters", "season": "Winter", "material": "Merino Wool",
            "cost_price": 38.0, "retail_price": 89.99, "launch_date": "2023-08-01",
            "base_demand": 18, "trend_type": "stable", "peak_months": [11, 12, 1, 2],
            "initial_stock": 140
        },
        {
            "sku": "KNT-003", "name": "Fine-Knit Buttoned Cardigan", "category": "Knitwear",
            "subcategory": "Cardigans", "season": "All-Season", "material": "Viscose Blend",
            "cost_price": 22.0, "retail_price": 54.99, "launch_date": "2023-01-20",
            "base_demand": 15, "trend_type": "stable", "peak_months": [3, 4, 9, 10],
            "initial_stock": 110
        },
        {
            "sku": "KNT-004", "name": "Ribbed Knit Polo Shirt", "category": "Knitwear",
            "subcategory": "Polos", "season": "Spring/Autumn", "material": "Organic Cotton",
            "cost_price": 18.0, "retail_price": 44.99, "launch_date": "2023-05-10",
            "base_demand": 11, "trend_type": "rising", "peak_months": [4, 5, 6, 9],
            "initial_stock": 20
        },
        {
            "sku": "KNT-005", "name": "Oversized Sweater Vest", "category": "Knitwear",
            "subcategory": "Vests", "season": "Spring/Autumn", "material": "Acrylic Blend",
            "cost_price": 15.0, "retail_price": 39.99, "launch_date": "2023-02-15",
            "base_demand": 6, "trend_type": "declining", "peak_months": [9, 10, 11],
            "initial_stock": 160
        },
        {
            "sku": "KNT-006", "name": "Mohair Open-Front Cardigan", "category": "Knitwear",
            "subcategory": "Cardigans", "season": "Winter", "material": "Mohair Blend",
            "cost_price": 32.0, "retail_price": 79.99, "launch_date": "2023-10-01",
            "base_demand": 9, "trend_type": "stable", "peak_months": [11, 12, 1],
            "initial_stock": 70
        },

        # --- Tops (Year-round & Summer peaks) ---
        {
            "sku": "TOP-001", "name": "Organic Heavyweight Cotton Tee", "category": "Tops",
            "subcategory": "T-Shirts", "season": "All-Season", "material": "100% Organic Cotton",
            "cost_price": 8.5, "retail_price": 24.99, "launch_date": "2023-01-01",
            "base_demand": 32, "trend_type": "rising", "peak_months": [5, 6, 7, 8],
            "initial_stock": 50
        },
        {
            "sku": "TOP-002", "name": "Tailored Oxford Button-Down", "category": "Tops",
            "subcategory": "Shirts", "season": "All-Season", "material": "Egyptian Cotton",
            "cost_price": 21.0, "retail_price": 59.99, "launch_date": "2023-01-15",
            "base_demand": 22, "trend_type": "stable", "peak_months": [3, 4, 9, 10],
            "initial_stock": 175
        },
        {
            "sku": "TOP-003", "name": "Pure Mulberry Silk Blouse", "category": "Tops",
            "subcategory": "Blouses", "season": "All-Season", "material": "Mulberry Silk",
            "cost_price": 36.0, "retail_price": 89.99, "launch_date": "2023-03-01",
            "base_demand": 12, "trend_type": "rising", "peak_months": [4, 5, 11, 12],
            "initial_stock": 18
        },
        {
            "sku": "TOP-004", "name": "Relaxed French Linen Shirt", "category": "Tops",
            "subcategory": "Shirts", "season": "Summer", "material": "100% French Linen",
            "cost_price": 24.0, "retail_price": 64.99, "launch_date": "2023-04-15",
            "base_demand": 26, "trend_type": "rising", "peak_months": [5, 6, 7, 8],
            "initial_stock": 40
        },
        {
            "sku": "TOP-005", "name": "Tie-Dye Graphic Sweatshirt", "category": "Tops",
            "subcategory": "Sweatshirts", "season": "All-Season", "material": "Cotton Fleece",
            "cost_price": 19.0, "retail_price": 49.99, "launch_date": "2023-02-01",
            "base_demand": 8, "trend_type": "declining", "peak_months": [1, 2, 3],
            "initial_stock": 190
        },
        {
            "sku": "TOP-006", "name": "Ribbed Seamless Tank Top", "category": "Tops",
            "subcategory": "Tanks", "season": "Summer", "material": "Nylon Elastane",
            "cost_price": 6.0, "retail_price": 19.99, "launch_date": "2023-05-01",
            "base_demand": 28, "trend_type": "stable", "peak_months": [6, 7, 8],
            "initial_stock": 220
        },

        # --- Bottoms ---
        {
            "sku": "BOT-001", "name": "High-Rise Straight Vintage Jeans", "category": "Bottoms",
            "subcategory": "Jeans", "season": "All-Season", "material": "Rigid Denim",
            "cost_price": 26.0, "retail_price": 69.99, "launch_date": "2023-01-01",
            "base_demand": 24, "trend_type": "rising", "peak_months": [3, 4, 8, 9, 10],
            "initial_stock": 35
        },
        {
            "sku": "BOT-002", "name": "Pleated Wide-Leg Trousers", "category": "Bottoms",
            "subcategory": "Trousers", "season": "All-Season", "material": "Polyester Wool Blend",
            "cost_price": 29.0, "retail_price": 74.99, "launch_date": "2023-02-10",
            "base_demand": 20, "trend_type": "rising", "peak_months": [3, 4, 9, 10, 11],
            "initial_stock": 28
        },
        {
            "sku": "BOT-003", "name": "Relaxed Cargo Utility Pants", "category": "Bottoms",
            "subcategory": "Pants", "season": "All-Season", "material": "Cotton Twill",
            "cost_price": 22.0, "retail_price": 59.99, "launch_date": "2023-03-01",
            "base_demand": 16, "trend_type": "declining", "peak_months": [4, 5, 6],
            "initial_stock": 165
        },
        {
            "sku": "BOT-004", "name": "High-Waist Tailored Linen Shorts", "category": "Bottoms",
            "subcategory": "Shorts", "season": "Summer", "material": "100% Linen",
            "cost_price": 16.0, "retail_price": 44.99, "launch_date": "2023-04-20",
            "base_demand": 19, "trend_type": "stable", "peak_months": [5, 6, 7, 8],
            "initial_stock": 130
        },
        {
            "sku": "BOT-005", "name": "Low-Rise Skinny Acid Jeans", "category": "Bottoms",
            "subcategory": "Jeans", "season": "All-Season", "material": "Stretch Denim",
            "cost_price": 20.0, "retail_price": 49.99, "launch_date": "2023-01-01",
            "base_demand": 5, "trend_type": "declining", "peak_months": [1, 2],
            "initial_stock": 240
        },
        {
            "sku": "BOT-006", "name": "A-Line Denim Midi Skirt", "category": "Bottoms",
            "subcategory": "Skirts", "season": "Spring/Autumn", "material": "Cotton Denim",
            "cost_price": 21.0, "retail_price": 54.99, "launch_date": "2023-03-15",
            "base_demand": 14, "trend_type": "rising", "peak_months": [4, 5, 6, 9],
            "initial_stock": 22
        },

        # --- Dresses ---
        {
            "sku": "DRS-001", "name": "Botanical Print Silk Wrap Dress", "category": "Dresses",
            "subcategory": "Midi Dresses", "season": "Summer", "material": "Silk Chiffon",
            "cost_price": 38.0, "retail_price": 99.99, "launch_date": "2023-04-01",
            "base_demand": 18, "trend_type": "rising", "peak_months": [5, 6, 7, 8],
            "initial_stock": 15
        },
        {
            "sku": "DRS-002", "name": "Minimalist Satin Slip Dress", "category": "Dresses",
            "subcategory": "Maxi Dresses", "season": "All-Season", "material": "Heavy Satin",
            "cost_price": 31.0, "retail_price": 79.99, "launch_date": "2023-02-14",
            "base_demand": 17, "trend_type": "rising", "peak_months": [5, 6, 11, 12],
            "initial_stock": 24
        },
        {
            "sku": "DRS-003", "name": "Tiered Bohemian Maxi Dress", "category": "Dresses",
            "subcategory": "Maxi Dresses", "season": "Summer", "material": "Cotton Voile",
            "cost_price": 27.0, "retail_price": 69.99, "launch_date": "2023-05-01",
            "base_demand": 15, "trend_type": "stable", "peak_months": [6, 7, 8],
            "initial_stock": 115
        },
        {
            "sku": "DRS-004", "name": "Velvet Draped Evening Gown", "category": "Dresses",
            "subcategory": "Evening Dresses", "season": "Winter", "material": "Stretch Velvet",
            "cost_price": 45.0, "retail_price": 129.99, "launch_date": "2023-10-01",
            "base_demand": 11, "trend_type": "stable", "peak_months": [11, 12],
            "initial_stock": 85
        },
        {
            "sku": "DRS-005", "name": "Neon Cut-Out Mini Party Dress", "category": "Dresses",
            "subcategory": "Mini Dresses", "season": "Summer", "material": "Polyester Spandex",
            "cost_price": 17.0, "retail_price": 44.99, "launch_date": "2023-03-01",
            "base_demand": 6, "trend_type": "declining", "peak_months": [6, 7],
            "initial_stock": 170
        },
        {
            "sku": "DRS-006", "name": "Ribbed Long-Sleeve Knit Midi Dress", "category": "Dresses",
            "subcategory": "Midi Dresses", "season": "Spring/Autumn", "material": "Ribbed Knit",
            "cost_price": 28.0, "retail_price": 69.99, "launch_date": "2023-08-15",
            "base_demand": 16, "trend_type": "rising", "peak_months": [9, 10, 11, 12],
            "initial_stock": 19
        }
    ]

def generate_synthetic_data(start_date="2024-01-01", end_date="2025-06-30", seed=42):
    """
    Generates synthetic daily sales records for all SKUs in the catalog.
    Includes:
      - Weekly seasonality (Friday-Sunday lifts)
      - Annual / Category seasonal peaks
      - Trend trajectory multipliers (rising = positive slope, declining = negative slope)
      - Promotional discount events
      - Poisson/Normal stochastic sales variance
    """
    np.random.seed(seed)
    random.seed(seed)

    products = get_product_catalog()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    sales_records = []
    product_records = []
    stock_records = []

    # Map day of week multiplier (0=Monday, 6=Sunday)
    dow_multipliers = {0: 0.85, 1: 0.88, 2: 0.92, 3: 0.98, 4: 1.25, 5: 1.45, 6: 1.30}

    # Pre-select random promotional discount days (Black Friday, Summer Sale, Mid-Season)
    promo_dates = {
        "2024-06-21": 0.20, "2024-06-22": 0.20, "2024-06-23": 0.20, # Summer Solstice Sale
        "2024-11-28": 0.30, "2024-11-29": 0.35, "2024-11-30": 0.30, # Black Friday Weekend
        "2024-12-01": 0.30, "2024-12-26": 0.25, "2024-12-27": 0.25, # Boxing Day
        "2025-05-23": 0.20, "2025-05-24": 0.20                       # Memorial Weekend
    }

    for p in products:
        sku = p["sku"]
        product_records.append({
            "sku": sku,
            "name": p["name"],
            "category": p["category"],
            "subcategory": p["subcategory"],
            "season": p["season"],
            "material": p["material"],
            "cost_price": p["cost_price"],
            "retail_price": p["retail_price"],
            "launch_date": p["launch_date"]
        })

        stock_records.append({
            "sku": sku,
            "current_stock": p["initial_stock"],
            "last_updated": end_date + " 23:59:59"
        })

        base_d = p["base_demand"]
        trend = p["trend_type"]
        peak_m = p["peak_months"]
        base_price = p["retail_price"]

        for d_idx in range(total_days):
            curr_date = start_dt + timedelta(days=d_idx)
            date_str = curr_date.strftime("%Y-%m-%d")
            month = curr_date.month
            dow = curr_date.weekday()

            # 1. Base demand with overall trend trajectory
            time_fraction = d_idx / total_days
            if trend == "rising":
                trend_factor = 0.70 + 0.65 * time_fraction  # +65% growth across timeframe
            elif trend == "declining":
                trend_factor = 1.30 - 0.70 * time_fraction  # -54% contraction across timeframe
            else:
                trend_factor = 1.00 + 0.05 * np.sin(2 * np.pi * time_fraction * 2)

            # 2. Seasonality factor
            season_factor = 1.60 if month in peak_m else 0.75

            # 3. Day of week factor
            dow_factor = dow_multipliers[dow]

            # 4. Promotions & Discounts
            discount = promo_dates.get(date_str, 0.0)
            # Occasional flash sale on select items
            if discount == 0.0 and random.random() < 0.03:
                discount = random.choice([0.10, 0.15])

            price_factor = 1.0 + (discount * 1.5) # price elasticity effect
            effective_unit_price = round(base_price * (1.0 - discount), 2)

            # 5. Composite expected demand lambda for Poisson sampling
            expected_demand = base_d * trend_factor * season_factor * dow_factor * price_factor
            noise = np.random.normal(0, max(1.0, expected_demand * 0.15))
            quantity_sold = max(0, int(np.round(np.random.poisson(max(0.1, expected_demand)) + noise)))

            sales_records.append({
                "date": date_str,
                "sku": sku,
                "quantity_sold": quantity_sold,
                "unit_price": effective_unit_price,
                "discount_applied": discount
            })

    df_sales = pd.DataFrame(sales_records)
    df_products = pd.DataFrame(product_records)
    df_stock = pd.DataFrame(stock_records)

    return df_sales, df_products, df_stock

def save_sample_datasets(output_dir="data/raw"):
    """Saves generated synthetic datasets to raw data directory."""
    os.makedirs(output_dir, exist_ok=True)
    df_sales, df_products, df_stock = generate_synthetic_data()

    sales_path = os.path.join(output_dir, "sales_data.csv")
    products_path = os.path.join(output_dir, "products.csv")
    stock_path = os.path.join(output_dir, "stock_levels.csv")

    df_sales.to_csv(sales_path, index=False)
    df_products.to_csv(products_path, index=False)
    df_stock.to_csv(stock_path, index=False)

    print(f"[OK] Generated {len(df_products)} products, {len(df_sales)} daily sales rows, and {len(df_stock)} stock records.")
    print(f"     Saved to: {sales_path}, {products_path}, {stock_path}")
    return df_sales, df_products, df_stock

if __name__ == "__main__":
    save_sample_datasets()
