"""Trend API Routes."""

from flask import Blueprint, jsonify
from backend.services.trend_service import TrendService

trend_bp = Blueprint("trend_bp", __name__)

@trend_bp.route("/api/trend/<sku>", methods=["GET"])
def get_trend(sku):
    """
    Endpoint: GET /api/trend/<sku>
    Returns trend status (rising/stable/declining), confidence score, and feature importance.
    """
    try:
        result = TrendService.get_trend_for_sku(sku=sku)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal classification error: {str(e)}"}), 500
