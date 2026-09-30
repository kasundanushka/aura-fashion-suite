# 👗 AURA: AI-Based Clothing Sales Forecasting & Fashion Trend Intelligence System

An academic full-stack prototype demonstrating three integrated Artificial Intelligence and Operations Research techniques for fashion retail inventory management:
1. **Time-Series Demand Forecasting**: Facebook Prophet with weekly/yearly seasonality and promotional pricing regressors, benchmarked against Ordinary Least Squares Linear Regression.
2. **Supervised Fashion Trend Classification**: Multi-class Random Forest Classifier predicting product trajectory (`rising`, `stable`, `declining`) using rolling momentum and category context, benchmarked against K-Nearest Neighbors (KNN).
3. **Rule-Based Inventory Optimisation**: Dynamic safety stock buffers, reorder points, and automated four-tier urgency alerts (`URGENT_REORDER`, `REORDER_SOON`, `SUFFICIENT_STOCK`, `OVERSTOCKED`).
4. **Interactive Single-Page Dashboard**: Vanilla HTML5/CSS3/ES6 JavaScript web interface powered by a Flask REST API backend and SQLite database.

---

## 📌 Proposal Objective Mapping Table

| Project Objective | Implementation Module | Description & Techniques |
|---|---|---|
| **1. Data Validation & Preprocessing** | [`src/data_pipeline/clean_data.py`](file:///d:/Ai%20project/src/data_pipeline/clean_data.py) | Schema validation, time-gap forward filling, zero negative tolerance, outlier logging |
| **2. Feature Engineering** | [`src/data_pipeline/feature_engineering.py`](file:///d:/Ai%20project/src/data_pipeline/feature_engineering.py) | Rolling statistics (7d, 14d, 28d), cyclical harmonics, category lag momentum, trend slope |
| **3. Time-Series Demand Forecasting** | [`src/forecasting/prophet_model.py`](file:///d:/Ai%20project/src/forecasting/prophet_model.py)<br>[`src/forecasting/baseline_regression.py`](file:///d:/Ai%20project/src/forecasting/baseline_regression.py) | Prophet additive model + 95% confidence intervals vs. Linear Regression baseline |
| **4. Fashion Trend Direction** | [`src/classification/trend_classifier.py`](file:///d:/Ai%20project/src/classification/trend_classifier.py)<br>[`src/classification/baseline_knn.py`](file:///d:/Ai%20project/src/classification/baseline_knn.py) | Random Forest with Gini feature importances vs. distance-weighted KNN baseline |
| **5. Inventory Reorder Engine** | [`src/optimisation/reorder_engine.py`](file:///d:/Ai%20project/src/optimisation/reorder_engine.py) | Statistical Safety Stock, Reorder Point, Trend Multipliers, and 4-tier Urgency Alert rules |
| **6. Centralized Model Evaluation** | [`src/evaluation/metrics.py`](file:///d:/Ai%20project/src/evaluation/metrics.py) | Standardized RMSE, MAPE, MAE, R², Accuracy, Macro F1, Confusion Matrix |
| **7. RESTful API Backend** | [`backend/app.py`](file:///d:/Ai%20project/backend/app.py)<br>[`backend/routes/`](file:///d:/Ai%20project/backend/routes/) | Flask REST service with CORS, error handlers, and CSV upload pipeline triggers |
| **8. Interactive Web UI** | [`frontend/index.html`](file:///d:/Ai%20project/frontend/index.html)<br>[`frontend/js/`](file:///d:/Ai%20project/frontend/js/) | Single-page dashboard, Chart.js interactive forecast curves, sortable alerts table |
| **9. Reproducible EDA & Modeling** | [`notebooks/`](file:///d:/Ai%20project/notebooks/) | Jupyter notebooks for EDA (`01_eda`), Forecasting (`02_forecasting`), and Classification (`03_trend`) |
| **10. Automated Test Suite** | [`tests/`](file:///d:/Ai%20project/tests/) | Full test coverage across data pipeline, forecasting, classification, optimisation, and API |

---

## 📐 Mathematical Formulations & Optimization Rules

### 1. Statistical Safety Stock ($SS$)
To ensure a 95% cycle service level ($Z = 1.65$) under supplier lead time $L$ (default 7 days) with daily demand standard deviation $\sigma_d$:
$$SS = \left\lceil Z \times \sigma_d \times \sqrt{L} \right\rceil$$

### 2. Reorder Point ($ROP$)
The inventory threshold triggering replenishment before stockout during lead time:
$$ROP = \left\lceil (\mu_d \times L) + SS \right\rceil$$
where $\mu_d$ is the mean daily forecasted demand.

### 3. Trend Multipliers ($M_{\text{trend}}$)
Proactive adjustments based on supervised machine learning trend classification:
* **Rising Trend**: $M_{\text{trend}} = 1.15$ (+15% demand buffer to capitalize on viral fashion momentum).
* **Stable Trend**: $M_{\text{trend}} = 1.00$ (Standard replenishment).
* **Declining Trend**: $M_{\text{trend}} = 0.80$ (-20% conservative dampening to avoid deadstock holding losses).

### 4. Net Recommended Reorder Quantity ($Q_{\text{reorder}}$)
$$Q_{\text{reorder}} = \max\left(0, \left\lceil (\hat{D}_{\text{horizon}} \times M_{\text{trend}}) + SS - I_{\text{current}} \right\rceil\right)$$

### 5. Urgency Classification Thresholds
* **`URGENT_REORDER`**: If $I_{\text{current}} \le 0.5 \times SS$ or current inventory is insufficient to cover lead time demand.
* **`REORDER_SOON`**: If $I_{\text{current}} \le ROP$.
* **`OVERSTOCKED`**: If $I_{\text{current}} > 2.5 \times (\hat{D}_{\text{horizon}} \times M_{\text{trend}} + SS)$.
* **`SUFFICIENT_STOCK`**: Stock is balanced between $ROP$ and the overstock threshold.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Setup Virtual Environment
```bash
# Navigate to project directory
cd "d:/Ai project"

# Create and activate virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset & Run Data Pipeline
```bash
# Step 1: Generate 18-month 30-SKU realistic dataset
python src/data_pipeline/generate_sample_data.py

# Step 2: Clean data, impute time gaps, and create feature matrix
python src/data_pipeline/clean_data.py
python src/data_pipeline/feature_engineering.py

# Step 3: Initialize and populate SQLite database
python database/init_db.py
```

### 3. Launch Flask Backend & Dashboard
```bash
python backend/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### 4. Run Automated Test Suite
```bash
python -m pytest -v
```

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Query Parameters / Payload |
|---|---|---|---|
| `GET` | `/api/health` | Service healthcheck | None |
| `GET` | `/api/products` | Returns all SKUs with current stock & category | None |
| `GET` | `/api/forecast/<sku>` | 4–12 week demand forecast with 95% confidence bounds & baseline comparison | `?weeks=8` (default: 8) |
| `GET` | `/api/trend/<sku>` | Trend status, class probabilities, and feature importances | None |
| `GET` | `/api/reorder-alerts` | All products ranked by urgency, recommended reorder quantities & budget | None |
| `POST` | `/api/upload-sales` | Ingests new sales CSV, validates schema, and triggers automated pipeline refresh | `multipart/form-data` with `file` |

---

## 🛡️ Non-Functional & Ethical Considerations
- **No PII**: All datasets contain strictly anonymized SKU codes and aggregate transactions; zero customer or payment details are collected or stored.
- **Input Validation**: All uploaded CSVs undergo schema verification, negative value rejection, and time gap imputation to prevent pipeline failure.
- **Defensive Error Handling**: Model predictions and database queries are wrapped with fallback algorithms ensuring uninterrupted API uptime.
- **Secrets Management**: Configuration defaults are isolated in `.env.example` and environment variables.
