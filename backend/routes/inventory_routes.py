"""Inventory & Data Upload Routes."""

import os
from flask import Blueprint, request, jsonify
from backend.services.inventory_service import InventoryService
from backend.services.pipeline_service import PipelineService
from database.init_db import get_db_connection

inventory_bp = Blueprint("inventory_bp", __name__)

@inventory_bp.route("/api/products", methods=["GET"])
def get_products():
    """
    Endpoint: GET /api/products
    Returns list of all registered SKUs, categories, and latest inventory stats.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
        SELECT p.sku, p.name, p.category, p.subcategory, p.season, p.material,
               p.cost_price, p.retail_price, p.launch_date,
               COALESCE(s.current_stock, 0) as current_stock
        FROM products p
        LEFT JOIN stock_levels s ON p.sku = s.sku
        ORDER BY p.category, p.sku
        """
        cursor.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify({"count": len(rows), "products": rows}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve products: {str(e)}"}), 500

@inventory_bp.route("/api/reorder-alerts", methods=["GET"])
def get_reorder_alerts():
    """
    Endpoint: GET /api/reorder-alerts
    Returns inventory status, trend context, safety stock, and recommended reorder quantities.
    """
    try:
        alerts = InventoryService.get_all_reorder_alerts()
        # Compute high-level KPI aggregations
        urgent_count = sum(1 for a in alerts if a["alert_level"] == "URGENT_REORDER")
        soon_count = sum(1 for a in alerts if a["alert_level"] == "REORDER_SOON")
        overstocked_count = sum(1 for a in alerts if a["alert_level"] == "OVERSTOCKED")
        sufficient_count = sum(1 for a in alerts if a["alert_level"] == "SUFFICIENT_STOCK")
        total_reorder_cost = sum(a.get("estimated_reorder_cost", 0) for a in alerts)

        return jsonify({
            "total_skus": len(alerts),
            "kpis": {
                "urgent_reorders": urgent_count,
                "reorder_soon": soon_count,
                "overstocked": overstocked_count,
                "sufficient_stock": sufficient_count,
                "total_estimated_reorder_budget": round(total_reorder_cost, 2)
            },
            "alerts": alerts
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to compute reorder alerts: {str(e)}"}), 500

@inventory_bp.route("/api/upload-sales", methods=["POST"])
def upload_sales():
    """
    Endpoint: POST /api/upload-sales
    Accepts multipart CSV file, validates columns, runs data pipeline, and updates models.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request. Use form-data with key 'file'."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV (.csv) files are supported."}), 400

        uploaded_by = request.form.get("uploaded_by") or request.headers.get("X-User-Email") or request.headers.get("X-Admin-Email") or "Store Buyer"
        res = PipelineService.process_sales_upload(file, uploaded_by=uploaded_by)
        return jsonify(res), 200

    except ValueError as ve:
        return jsonify({"error": f"Data validation error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Pipeline execution failed: {str(e)}"}), 500
