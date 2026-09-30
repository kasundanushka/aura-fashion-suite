"""
Main Flask Server Entrypoint for Clothing AI System.
Configures CORS, registers API route blueprints, serves dashboard frontend,
and provides robust error handlers.
"""

import os
import sys
import logging
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.routes.forecast_routes import forecast_bp
from backend.routes.trend_routes import trend_bp
from backend.routes.inventory_routes import inventory_bp
from backend.routes.auth_routes import auth_bp
from backend.routes.admin_routes import admin_bp
from backend.routes.buyer_routes import buyer_bp
from database.init_db import populate_database_from_csv, init_upload_history_table, get_db_connection, DB_PATH

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClothingAIBackend")

def create_app():
    """Application factory for Flask server."""
    app = Flask(
        __name__,
        static_folder="../frontend",
        static_url_path=""
    )
    
    # Enable Cross-Origin Resource Sharing
    CORS(app)

    # Register Blueprints
    app.register_blueprint(forecast_bp)
    app.register_blueprint(trend_bp)
    app.register_blueprint(inventory_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(buyer_bp)

    # Ensure database is initialized and upload history table exists
    if not os.path.exists(DB_PATH):
        logger.info("Initializing SQLite database on startup...")
        populate_database_from_csv()
    else:
        try:
            conn = get_db_connection()
            init_upload_history_table(conn)
            conn.close()
        except Exception as err:
            logger.warning(f"Could not verify upload history table: {err}")

    # Serve Frontend Single Page App
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/admin")
    @app.route("/admin.html")
    def admin_page():
        return send_from_directory(app.static_folder, "admin.html")

    @app.route("/dashboard")
    @app.route("/dashboard.html")
    def dashboard_page():
        return send_from_directory(app.static_folder, "dashboard.html")

    @app.route("/buyer")
    @app.route("/buyer.html")
    def buyer_page():
        return send_from_directory(app.static_folder, "buyer.html")

    # API Healthcheck endpoint
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "Clothing AI Inventory Optimization API",
            "version": "1.0.0"
        }), 200

    # Global Error Handlers
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify({"error": "Resource not found", "details": str(e)}), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({"error": "Internal server error occurred. Check server logs."}), 500

    return app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]

    logger.info(f"Starting Clothing AI Backend server at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
