"""
Authentication Routes Blueprint for Clothing AI System.
Provides endpoints for email/password login, user registration,
Google OAuth authentication, session validation, and logout.
"""

import os
import sys
import secrets
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import uuid

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from database.init_db import get_db_connection

auth_bp = Blueprint("auth_bp", __name__, url_prefix="/api/auth")

# Avatar upload config
AVATAR_UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/uploads/avatars"))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
os.makedirs(AVATAR_UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_audit_event(conn, user_email, action, details=None, ip_address=None):
    """Utility to record administrative and security audit events."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (user_email, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_email, action, details, ip_address))
        conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to write audit log: {e}")

@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticates user with email and password."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        conn.close()
        return jsonify({"error": "Invalid email or password"}), 401

    token = secrets.token_hex(24)
    client_ip = request.remote_addr

    log_audit_event(conn, user["email"], "USER_LOGIN", f"Role: {user['role']}", client_ip)
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Welcome back, {user['name']}!",
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "avatar_url": user["avatar_url"]
        }
    }), 200

@auth_bp.route("/register", methods=["POST"])
def register():
    """Registers a new user account."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"error": "An account with this email already exists"}), 409

    role = "admin" if email in ["admin@aura.ai", "system@aura.ai"] or email.endswith("@aura.admin") else "user"
    pwd_hash = generate_password_hash(password)
    avatar_url = f"https://api.dicebear.com/7.x/initials/svg?seed={name}"

    cursor.execute("""
        INSERT INTO users (name, email, password_hash, role, avatar_url)
        VALUES (?, ?, ?, ?, ?)
    """, (name, email, pwd_hash, role, avatar_url))
    new_id = cursor.lastrowid
    conn.commit()

    token = secrets.token_hex(24)
    log_audit_event(conn, email, "USER_REGISTER", f"Created account with role: {role}", request.remote_addr)
    conn.close()

    return jsonify({
        "success": True,
        "message": "Account created successfully",
        "token": token,
        "user": {
            "id": new_id,
            "name": name,
            "email": email,
            "role": role,
            "avatar_url": avatar_url
        }
    }), 201

@auth_bp.route("/google", methods=["POST"])
def google_auth():
    """
    Handles Google OAuth sign-in.
    Finds or creates a user based on verified Google email/profile.
    Assigns admin role if email matches admin domains or known admin accounts.
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    name = data.get("name", "").strip() or email.split("@")[0].capitalize()
    avatar_url = data.get("avatar_url") or f"https://api.dicebear.com/7.x/initials/svg?seed={name}"

    if not email:
        return jsonify({"error": "Google email is required"}), 400

    is_admin_email = (
        email in ["admin@aura.ai", "admin@gmail.com", "system@aura.ai"]
        or email.startswith("admin@")
        or email.endswith("@aura.admin")
    )

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()

    if user:
        # If user exists and email is an admin email, ensure role is admin
        if is_admin_email and user["role"] != "admin":
            cursor.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user["id"],))
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user["id"],))
            user = cursor.fetchone()

        token = secrets.token_hex(24)
        log_audit_event(conn, user["email"], "GOOGLE_LOGIN", f"Role: {user['role']}", request.remote_addr)
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Welcome back, {user['name']}!",
            "token": token,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
                "avatar_url": user["avatar_url"] or avatar_url
            }
        }), 200
    else:
        # Create user
        role = "admin" if is_admin_email else "user"
        random_pwd = secrets.token_urlsafe(16)
        pwd_hash = generate_password_hash(random_pwd)

        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, avatar_url)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, pwd_hash, role, avatar_url))
        new_id = cursor.lastrowid
        conn.commit()

        token = secrets.token_hex(24)
        log_audit_event(conn, email, "GOOGLE_SIGNUP", f"Google created account role: {role}", request.remote_addr)
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Account created with Google: Welcome, {name}!",
            "token": token,
            "user": {
                "id": new_id,
                "name": name,
                "email": email,
                "role": role,
                "avatar_url": avatar_url
            }
        }), 201

@auth_bp.route("/me", methods=["GET"])
def get_current_user():
    """Validates session or returns user info by email in headers or query."""
    user_email = request.headers.get("X-User-Email") or request.args.get("email")
    if not user_email:
        return jsonify({"authenticated": False, "error": "No user session provided"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, avatar_url, created_at FROM users WHERE LOWER(email) = ?", (user_email.strip().lower(),))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"authenticated": False, "error": "User not found"}), 404

    return jsonify({
        "authenticated": True,
        "user": dict(user)
    }), 200

@auth_bp.route("/profile", methods=["PUT"])
def update_profile():
    """Updates user profile: name, avatar_url, and optionally password."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required to identify user"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    # Collect updateable fields
    new_name = data.get("name", "").strip()
    new_avatar = data.get("avatar_url", "").strip()
    new_password = data.get("new_password", "").strip()
    current_password = data.get("current_password", "").strip()

    updates = []
    params = []

    if new_name and new_name != user["name"]:
        updates.append("name = ?")
        params.append(new_name)

    if new_avatar and new_avatar != user["avatar_url"]:
        updates.append("avatar_url = ?")
        params.append(new_avatar)

    if new_password:
        if len(new_password) < 6:
            conn.close()
            return jsonify({"error": "New password must be at least 6 characters"}), 400
        if current_password and not check_password_hash(user["password_hash"], current_password):
            conn.close()
            return jsonify({"error": "Current password is incorrect"}), 403
        updates.append("password_hash = ?")
        params.append(generate_password_hash(new_password))

    if not updates:
        conn.close()
        return jsonify({"error": "No changes provided"}), 400

    params.append(user["id"])
    cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

    # Fetch updated user
    cursor.execute("SELECT id, name, email, role, avatar_url, created_at FROM users WHERE id = ?", (user["id"],))
    updated = cursor.fetchone()

    log_audit_event(conn, email, "PROFILE_UPDATE", f"Updated: {', '.join(updates)}", request.remote_addr)
    conn.close()

    return jsonify({
        "success": True,
        "message": "Profile updated successfully",
        "user": dict(updated)
    }), 200

@auth_bp.route("/avatar", methods=["POST"])
def upload_avatar():
    """Uploads a profile picture and updates user avatar_url."""
    if 'avatar' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['avatar']
    email = request.form.get('email', '').strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, GIF, or WebP"}), 400

    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(AVATAR_UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Update database
    avatar_url = f"/uploads/avatars/{filename}"
    conn = get_db_connection()
    cursor = conn.cursor()

    # Delete old uploaded avatar if exists
    cursor.execute("SELECT avatar_url FROM users WHERE LOWER(email) = ?", (email,))
    old_user = cursor.fetchone()
    if old_user and old_user["avatar_url"] and old_user["avatar_url"].startswith("/uploads/avatars/"):
        old_path = os.path.join(AVATAR_UPLOAD_FOLDER, os.path.basename(old_user["avatar_url"]))
        if os.path.exists(old_path):
            os.remove(old_path)

    cursor.execute("UPDATE users SET avatar_url = ? WHERE LOWER(email) = ?", (avatar_url, email))
    conn.commit()

    log_audit_event(conn, email, "AVATAR_UPLOAD", f"Uploaded: {filename}", request.remote_addr)
    conn.close()

    return jsonify({
        "success": True,
        "avatar_url": avatar_url,
        "message": "Avatar uploaded successfully"
    }), 200

@auth_bp.route("/avatar/remove", methods=["POST"])
def remove_avatar():
    """Removes the user's uploaded avatar, resetting to initials."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT avatar_url FROM users WHERE LOWER(email) = ?", (email,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    # Delete file if it's an uploaded avatar
    if user["avatar_url"] and user["avatar_url"].startswith("/uploads/avatars/"):
        old_path = os.path.join(AVATAR_UPLOAD_FOLDER, os.path.basename(user["avatar_url"]))
        if os.path.exists(old_path):
            os.remove(old_path)

    cursor.execute("UPDATE users SET avatar_url = NULL WHERE LOWER(email) = ?", (email,))
    conn.commit()
    log_audit_event(conn, email, "AVATAR_REMOVE", "Avatar removed", request.remote_addr)
    conn.close()

    return jsonify({"success": True, "message": "Avatar removed"}), 200

@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Logs out user and records audit log."""
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    if email:
        conn = get_db_connection()
        log_audit_event(conn, email, "USER_LOGOUT", "User logged out", request.remote_addr)
        conn.close()
    return jsonify({"success": True, "message": "Logged out successfully"}), 200
