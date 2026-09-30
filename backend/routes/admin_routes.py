"""
Admin Routes Blueprint for Clothing AI System.
Provides administrative endpoints with strict role-based access control.
Only users with role == 'admin' can access these endpoints.
"""

import os
import sys
from functools import wraps
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.security import generate_password_hash

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from database.init_db import get_db_connection
from backend.routes.auth_routes import log_audit_event

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/api/admin")

def admin_required(f):
    """Decorator to enforce admin role verification on routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check header for user email or admin token
        admin_email = request.headers.get("X-Admin-Email") or request.headers.get("X-User-Email")
        
        # Also check JSON body or query parameters
        if not admin_email and request.is_json:
            admin_email = (request.get_json(silent=True) or {}).get("admin_email")
        if not admin_email:
            admin_email = request.args.get("admin_email")

        if not admin_email:
            return jsonify({
                "error": "Admin authentication required. Missing credentials.",
                "code": "UNAUTHORIZED"
            }), 401

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, role FROM users WHERE LOWER(email) = ?", (admin_email.strip().lower(),))
        user = cursor.fetchone()
        conn.close()

        if not user or user["role"] != "admin":
            return jsonify({
                "error": "Access Denied. Only administrators are authorized to access this panel.",
                "code": "FORBIDDEN"
            }), 403

        # Store authenticated admin on request context
        request.current_admin = dict(user)
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route("/verify", methods=["GET", "POST"])
@admin_required
def verify_admin():
    """Confirms current user has valid admin permissions."""
    return jsonify({
        "success": True,
        "admin": request.current_admin,
        "message": "Admin privileges verified."
    }), 200

@admin_bp.route("/stats", methods=["GET"])
@admin_required
def get_admin_stats():
    """Returns comprehensive system statistics and health indicators."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM sales_history")
    total_sales_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    admin_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reorder_recommendations WHERE alert_level = 'URGENT_REORDER'")
    urgent_alerts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM forecasts")
    forecasts_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audit_logs")
    total_logs = cursor.fetchone()[0]

    try:
        cursor.execute("SELECT COUNT(*) FROM csv_upload_history")
        total_csv_uploads = cursor.fetchone()[0]
    except Exception:
        total_csv_uploads = 0

    conn.close()

    return jsonify({
        "success": True,
        "stats": {
            "total_products": total_products,
            "total_sales_records": total_sales_records,
            "total_users": total_users,
            "admin_users": admin_users,
            "urgent_alerts": urgent_alerts,
            "forecasts_count": forecasts_count,
            "total_logs": total_logs,
            "total_csv_uploads": total_csv_uploads,
            "backend_version": "1.0.0",
            "db_status": "Online & Operational",
            "ml_models": ["Prophet v1.1.5", "RandomForestClassifier", "EOQ Engine"]
        }
    }), 200

@admin_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    """Lists all registered users and their roles."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, avatar_url, created_at FROM users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    users = [dict(row) for row in rows]
    return jsonify({"success": True, "users": users}), 200

@admin_bp.route("/users", methods=["POST"])
@admin_required
def create_user():
    """Allows an admin to manually provision a user or administrator."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    role = data.get("role", "user").strip().lower()

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400

    if role not in ["user", "admin"]:
        return jsonify({"error": "Invalid role specified. Must be 'user' or 'admin'."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"error": f"A user with email '{email}' already exists."}), 409

    pwd_hash = generate_password_hash(password)
    avatar = f"https://api.dicebear.com/7.x/initials/svg?seed={name}"

    cursor.execute("""
        INSERT INTO users (name, email, password_hash, role, avatar_url)
        VALUES (?, ?, ?, ?, ?)
    """, (name, email, pwd_hash, role, avatar))
    new_id = cursor.lastrowid
    conn.commit()

    log_audit_event(conn, request.current_admin["email"], "ADMIN_CREATE_USER", f"Created {email} as {role}", request.remote_addr)
    conn.close()

    return jsonify({
        "success": True,
        "message": f"User '{name}' created with role '{role}'.",
        "user": {
            "id": new_id,
            "name": name,
            "email": email,
            "role": role,
            "avatar_url": avatar
        }
    }), 201

@admin_bp.route("/users/<int:user_id>/role", methods=["PATCH"])
@admin_required
def update_user_role(user_id):
    """Allows promoting or demoting user roles."""
    data = request.get_json(silent=True) or {}
    new_role = data.get("role", "").strip().lower()

    if new_role not in ["admin", "user"]:
        return jsonify({"error": "Role must be 'admin' or 'user'."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, role FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if not target_user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    # Prevent demoting the root administrator
    if target_user["email"].lower() == "admin@aura.ai" and new_role != "admin":
        conn.close()
        return jsonify({"error": "The primary system administrator cannot be demoted."}), 403

    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()

    log_audit_event(
        conn,
        request.current_admin["email"],
        "ADMIN_UPDATE_ROLE",
        f"Changed {target_user['email']} role from {target_user['role']} to {new_role}",
        request.remote_addr
    )
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Updated {target_user['email']} role to {new_role}."
    }), 200

@admin_bp.route("/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    """Allows an administrator to delete a user account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if not target_user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    if target_user["email"].lower() == "admin@aura.ai":
        conn.close()
        return jsonify({"error": "The primary system administrator cannot be deleted."}), 403

    if target_user["id"] == request.current_admin["id"]:
        conn.close()
        return jsonify({"error": "You cannot delete your own active administrator account."}), 403

    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()

    log_audit_event(
        conn,
        request.current_admin["email"],
        "ADMIN_DELETE_USER",
        f"Deleted user {target_user['email']} (ID {user_id})",
        request.remote_addr
    )
    conn.close()

    return jsonify({"success": True, "message": f"User {target_user['email']} deleted."}), 200

@admin_bp.route("/logs", methods=["GET"])
@admin_required
def get_audit_logs():
    """Returns the latest 50 administrative and user audit logs."""
    limit = min(int(request.args.get("limit", 50)), 100)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_email, action, details, ip_address, created_at
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    logs = [dict(row) for row in rows]
    return jsonify({"success": True, "logs": logs}), 200

@admin_bp.route("/products", methods=["POST"])
@admin_required
def add_product():
    """Allows adding a new SKU and stock level directly from the admin panel."""
    data = request.get_json(silent=True) or {}
    sku = data.get("sku", "").strip().upper()
    name = data.get("name", "").strip()
    category = data.get("category", "").strip()
    subcategory = data.get("subcategory", "").strip()
    season = data.get("season", "All-Season").strip()
    material = data.get("material", "Cotton Blend").strip()
    cost_price = float(data.get("cost_price", 25.0))
    retail_price = float(data.get("retail_price", 59.99))
    initial_stock = int(data.get("current_stock", 100))

    if not sku or not name or not category:
        return jsonify({"error": "SKU, Name, and Category are required."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sku FROM products WHERE sku = ?", (sku,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"error": f"Product with SKU '{sku}' already exists."}), 409

    cursor.execute("""
        INSERT INTO products (sku, name, category, subcategory, season, material, cost_price, retail_price, launch_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, DATE('now'))
    """, (sku, name, category, subcategory, season, material, cost_price, retail_price))

    # Also insert or replace stock level
    cursor.execute("""
        INSERT OR REPLACE INTO stock_levels (sku, current_stock)
        VALUES (?, ?)
    """, (sku, initial_stock))

    conn.commit()

    log_audit_event(
        conn,
        request.current_admin["email"],
        "ADMIN_ADD_PRODUCT",
        f"Added SKU {sku} ({name}) with stock {initial_stock}",
        request.remote_addr
    )
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Product [{sku}] {name} added successfully.",
        "sku": sku
    }), 201

@admin_bp.route("/pipeline/retrain", methods=["POST"])
@admin_required
def trigger_retrain():
    """Triggers real execution of data cleaning, feature engineering, and model refresh."""
    try:
        from backend.services.pipeline_service import PipelineService
        result = PipelineService.trigger_full_pipeline_refresh()

        conn = get_db_connection()
        log_audit_event(
            conn,
            request.current_admin["email"],
            "ADMIN_TRIGGER_PIPELINE",
            f"Executed full AI pipeline refresh (Found {result.get('urgent_alerts_count', 0)} urgent alerts)",
            request.remote_addr
        )
        conn.close()

        return jsonify({
            "success": True,
            "message": result["message"],
            "urgent_alerts": result.get("urgent_alerts_count", 0),
            "status": "completed"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

@admin_bp.route("/export/<dataset_name>", methods=["GET"])
@admin_required
def export_dataset(dataset_name):
    """
    Exports processed datasets or live inventory intelligence as downloadable CSV files.
    Supported datasets: clean_sales, clean_products, feature_matrix, reorder_recommendations
    """
    import io
    import pandas as pd
    from flask import Response

    valid_file_exports = {
        "clean_sales": "data/processed/clean_sales.csv",
        "clean_products": "data/processed/clean_products.csv",
        "feature_matrix": "data/processed/feature_matrix.csv"
    }

    if dataset_name in valid_file_exports:
        path = valid_file_exports[dataset_name]
        if os.path.exists(path):
            directory = os.path.abspath(os.path.dirname(path))
            filename = os.path.basename(path)
            return send_from_directory(directory, filename, as_attachment=True, download_name=f"{dataset_name}.csv")
        else:
            return jsonify({"error": f"Processed dataset '{dataset_name}' not found. Run pipeline first."}), 404

    elif dataset_name == "reorder_recommendations":
        try:
            conn = get_db_connection()
            query = """
                SELECT r.sku, p.name, p.category, r.alert_level, r.recommended_quantity,
                       r.current_stock, r.safety_stock, r.forecasted_demand, r.trend_factor,
                       p.cost_price, p.retail_price,
                       ROUND(r.recommended_quantity * p.cost_price, 2) as estimated_reorder_cost,
                       r.generated_at
                FROM reorder_recommendations r
                LEFT JOIN products p ON r.sku = p.sku
                ORDER BY r.alert_level, r.recommended_quantity DESC
            """
            df_reorder = pd.read_sql_query(query, conn)
            conn.close()

            csv_buffer = io.StringIO()
            df_reorder.to_csv(csv_buffer, index=False)
            csv_output = csv_buffer.getvalue()

            return Response(
                csv_output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=reorder_recommendations.csv"}
            )
        except Exception as err:
            return jsonify({"error": f"Failed to export recommendations: {err}"}), 500

    return jsonify({"error": f"Invalid dataset name '{dataset_name}'. Allowed: clean_sales, clean_products, feature_matrix, reorder_recommendations"}), 400

@admin_bp.route("/maintenance/vacuum", methods=["POST"])
@admin_required
def maintenance_vacuum():
    """Compacts SQLite storage and performs database integrity checks."""
    try:
        from database.init_db import DB_PATH
        before_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]

        cursor.execute("VACUUM")
        conn.commit()
        conn.close()

        after_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        saved_bytes = max(0, before_size - after_size)

        log_conn = get_db_connection()
        log_audit_event(
            log_conn,
            request.current_admin["email"],
            "ADMIN_DATABASE_VACUUM",
            f"Integrity: {integrity}, Saved: {saved_bytes} bytes",
            request.remote_addr
        )
        log_conn.close()

        return jsonify({
            "success": True,
            "integrity_check": integrity,
            "before_size_bytes": before_size,
            "after_size_bytes": after_size,
            "reclaimed_bytes": saved_bytes,
            "message": f"Database optimized. Integrity status: {integrity}."
        }), 200
    except Exception as err:
        return jsonify({"error": f"Maintenance failed: {str(err)}"}), 500

@admin_bp.route("/upload-history", methods=["GET"])
@admin_required
def get_upload_history():
    """Returns the full history of all CSV files uploaded into the system."""
    limit = min(int(request.args.get("limit", 100)), 200)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, filename, saved_path, uploaded_by, file_size_bytes, row_count,
                   columns_detected, status, pipeline_status, error_message, uploaded_at
            FROM csv_upload_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
    except Exception as e:
        history = []
    finally:
        conn.close()

    return jsonify({"success": True, "history": history}), 200

@admin_bp.route("/upload-csv", methods=["POST"])
@admin_required
def admin_upload_csv():
    """Allows an administrator to upload a new sales CSV directly from the Admin Panel."""
    if "file" not in request.files:
        return jsonify({"error": "No CSV file provided. Please choose a file."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV (.csv) files are supported."}), 400

    try:
        from backend.services.pipeline_service import PipelineService
        admin_email = request.current_admin.get("email", "System Admin")
        res = PipelineService.process_sales_upload(file, uploaded_by=admin_email)

        conn = get_db_connection()
        log_audit_event(
            conn,
            admin_email,
            "ADMIN_UPLOAD_CSV",
            f"Uploaded dataset '{file.filename}' ({res.get('row_count', 0)} rows)",
            request.remote_addr
        )
        conn.close()

        return jsonify(res), 200
    except ValueError as ve:
        return jsonify({"error": f"Validation error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Upload processing failed: {str(e)}"}), 500

@admin_bp.route("/upload-history/<int:upload_id>/download", methods=["GET"])
@admin_required
def download_uploaded_csv(upload_id):
    """Allows an administrator to download an archived uploaded CSV file."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, saved_path FROM csv_upload_history WHERE id = ?", (upload_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Record not found."}), 404

    target_path = row["saved_path"]
    download_name = row["filename"] or "dataset.csv"

    if target_path and os.path.exists(target_path):
        directory = os.path.abspath(os.path.dirname(target_path))
        filename = os.path.basename(target_path)
        return send_from_directory(directory, filename, as_attachment=True, download_name=download_name)

    # Fallback to data/raw/sales_data.csv if available
    raw_path = os.path.abspath("data/raw/sales_data.csv")
    if os.path.exists(raw_path):
        return send_from_directory(os.path.dirname(raw_path), "sales_data.csv", as_attachment=True, download_name=download_name)

    return jsonify({"error": "Physical CSV file no longer exists on server storage."}), 404

@admin_bp.route("/upload-history/<int:upload_id>", methods=["DELETE"])
@admin_required
def delete_upload_record(upload_id):
    """Allows an administrator to remove an entry from the upload history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM csv_upload_history WHERE id = ?", (upload_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Record not found."}), 404

    cursor.execute("DELETE FROM csv_upload_history WHERE id = ?", (upload_id,))
    conn.commit()
    log_audit_event(
        conn,
        request.current_admin["email"],
        "ADMIN_DELETE_UPLOAD_RECORD",
        f"Deleted upload history record #{upload_id} ({row['filename']})",
        request.remote_addr
    )
    conn.close()
    return jsonify({"success": True, "message": f"Record #{upload_id} deleted."}), 200
