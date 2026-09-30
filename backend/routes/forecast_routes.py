"""Forecast API Routes."""

from flask import Blueprint, request, jsonify
from backend.services.forecast_service import ForecastService

forecast_bp = Blueprint("forecast_bp", __name__)

@forecast_bp.route("/api/forecast/<sku>", methods=["GET"])
def get_forecast(sku):
    """
    Endpoint: GET /api/forecast/<sku>?weeks=8
    Returns historical daily demand + future predicted demand with confidence intervals.
    """
    try:
        weeks = int(request.args.get("weeks", 8))
        if weeks < 1 or weeks > 52:
            return jsonify({"error": "Query parameter 'weeks' must be between 1 and 52"}), 400

        result = ForecastService.get_forecast_for_sku(sku=sku, weeks=weeks)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal forecasting error: {str(e)}"}), 500
