"""
Buyer Routes Blueprint for Clothing AI System.
Provides endpoints for saving/retrieving body measurements
and uploading clothing images for virtual try-on.
"""

import os
import sys
import uuid
import logging
import shutil
import concurrent.futures
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from database.init_db import get_db_connection

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

logger = logging.getLogger("BuyerRoutes")

buyer_bp = Blueprint("buyer_bp", __name__, url_prefix="/api/buyer")

# Upload configs
CLOTH_UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/uploads/clothes"))
PERSON_UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/uploads/persons"))
TRYON_OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/uploads/tryon"))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
os.makedirs(CLOTH_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSON_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRYON_OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_buyer_measurements_table():
    """Create buyer_measurements table if it does not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
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
        )
    """)
    conn.commit()
    conn.close()


# Initialize table on module load
ensure_buyer_measurements_table()


@buyer_bp.route("/measurements", methods=["POST"])
def save_measurements():
    """Save or update buyer's gender and body measurements."""
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    gender = data.get("gender", "").strip().lower()

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if gender not in ("male", "female"):
        return jsonify({"error": "gender must be 'male' or 'female'"}), 400

    height = data.get("height")
    chest = data.get("chest")
    waist = data.get("waist")
    hips = data.get("hips")
    shoulder_width = data.get("shoulder")
    arm_length = data.get("arm")
    leg_length = data.get("leg")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if measurements already exist for this user
    cursor.execute("SELECT id FROM buyer_measurements WHERE user_id = ?", (user_id,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE buyer_measurements
            SET gender=?, height=?, chest=?, waist=?, hips=?,
                shoulder_width=?, arm_length=?, leg_length=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE user_id=?
        """, (gender, height, chest, waist, hips,
              shoulder_width, arm_length, leg_length, user_id))
    else:
        cursor.execute("""
            INSERT INTO buyer_measurements
            (user_id, gender, height, chest, waist, hips, shoulder_width, arm_length, leg_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, gender, height, chest, waist, hips,
              shoulder_width, arm_length, leg_length))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Measurements saved successfully"
    }), 200


@buyer_bp.route("/measurements", methods=["GET"])
def get_measurements():
    """Retrieve buyer's saved measurements."""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id query parameter is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM buyer_measurements WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"found": False, "message": "No measurements found"}), 200

    return jsonify({
        "found": True,
        "measurements": {
            "gender": row["gender"],
            "height": row["height"],
            "chest": row["chest"],
            "waist": row["waist"],
            "hips": row["hips"],
            "shoulder_width": row["shoulder_width"],
            "arm_length": row["arm_length"],
            "leg_length": row["leg_length"],
            "updated_at": row["updated_at"]
        }
    }), 200


from backend.services.garment_processor import process_garment_image


@buyer_bp.route("/upload-cloth", methods=["POST"])
def upload_cloth():
    """Upload a clothing image for virtual try-on, automatically detecting and segmenting front and back."""
    if 'cloth' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['cloth']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, GIF, or WebP"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    base_id = uuid.uuid4().hex[:10]
    filename = f"{base_id}.{ext}"
    filepath = os.path.join(CLOTH_UPLOAD_FOLDER, filename)
    file.save(filepath)

    raw_cloth_url = f"/uploads/clothes/{filename}"

    # Process garment for background segmentation and front/back detection
    try:
        proc = process_garment_image(filepath, filename_base=f"proc_{base_id}")
    except Exception as e:
        print(f"Error processing garment image: {e}")
        proc = {
            "front_url": raw_cloth_url,
            "back_url": raw_cloth_url,
            "has_back": False,
            "auto_generated_back": True,
            "palette": ["#ffffff", "#1a3a6b"],
            "primary_color": "#ffffff",
            "secondary_color": "#1a3a6b"
        }

    return jsonify({
        "success": True,
        "cloth_url": raw_cloth_url,
        "front_url": proc["front_url"],
        "back_url": proc["back_url"],
        "has_back": proc["has_back"],
        "auto_generated_back": proc["auto_generated_back"],
        "palette": proc["palette"],
        "primary_color": proc["primary_color"],
        "secondary_color": proc["secondary_color"],
        "category": proc.get("category", "top"),
        "category_name": proc.get("category_name", "Shirt / Top"),
        "category_icon": proc.get("category_icon", "👕"),
        "category_label_si": proc.get("category_label_si", "ෂර්ට් / Top"),
        "target_area": proc.get("target_area", "Upper Torso & Shoulders"),
        "model_male_front": proc.get("model_male_front", "/uploads/clothes/model_male_front.jpg"),
        "model_female_front": proc.get("model_female_front", "/uploads/clothes/model_female_front.jpg"),
        "model_back": proc.get("model_back", "/uploads/clothes/model_back.jpg"),
        "message": "Clothing image analyzed and processed successfully"
    }), 200


def _remove_or_soften_background(img_rgba):
    """
    Intelligently detects background color and removes it, creating clean transparency.
    """
    extrema = img_rgba.getextrema()
    # If image already contains strong transparency
    if extrema[3][0] < 200:
        return img_rgba

    # Sample borders to identify background tint
    w, h = img_rgba.size
    sample_points = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (0, h // 2), (w - 1, h // 2)
    ]
    bg_r = sum(img_rgba.getpixel(p)[0] for p in sample_points) / len(sample_points)
    bg_g = sum(img_rgba.getpixel(p)[1] for p in sample_points) / len(sample_points)
    bg_b = sum(img_rgba.getpixel(p)[2] for p in sample_points) / len(sample_points)

    data = img_rgba.getdata()
    new_pixels = []
    for p in data:
        r, g, b, a = p[0], p[1], p[2], p[3] if len(p) > 3 else 255
        dist = ((r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2) ** 0.5
        if dist < 32:
            new_pixels.append((r, g, b, 0))
        elif dist < 58:
            alpha = int(255 * (dist - 32) / 26)
            new_pixels.append((r, g, b, min(a, alpha)))
        else:
            new_pixels.append((r, g, b, a))

    res = Image.new("RGBA", img_rgba.size)
    res.putdata(new_pixels)
    return res


def _find_torso_anchor(person_rgb):
    """
    Finds optimal (pos_x, pos_y, target_w) for garment on any person photo (full-body or portrait).
    Uses skin color & contour analysis to detect head/face and shoulders.
    """
    arr = np.array(person_rgb)
    h, w, _ = arr.shape

    # RGB to YCrCb for skin color detection
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    y_val = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y_val) * 0.713 + 128
    cb = (b - y_val) * 0.564 + 128

    skin = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 128) & (y_val > 40)

    # Search upper 65% for face
    upper_skin = skin[:int(h * 0.65), int(w * 0.15):int(w * 0.85)]
    ys, xs = np.where(upper_skin)

    if len(ys) > 60:
        face_min_y = ys.min()
        face_max_y = ys.max()
        face_min_x = xs.min() + int(w * 0.15)
        face_max_x = xs.max() + int(w * 0.15)

        face_h = face_max_y - face_min_y
        face_w = max(35, face_max_x - face_min_x)
        face_cx = (face_min_x + face_max_x) // 2

        # Garment collar starts right at the base of the neck
        chin_y = face_min_y + int(face_h * 0.88)
        pos_y = min(int(h * 0.68), chin_y + int(face_h * 0.15))

        # Shoulders width ~ 2.4x to 3.0x face width
        target_w = int(min(w * 0.90, max(w * 0.50, face_w * 2.6)))
        pos_x = max(0, min(w - target_w, face_cx - target_w // 2))
        return int(pos_x), int(pos_y), int(target_w)
    else:
        aspect = h / float(w)
        if aspect < 1.3:
            pos_y = int(h * 0.40)
            target_w = int(w * 0.68)
        else:
            pos_y = int(h * 0.28)
            target_w = int(w * 0.58)
        pos_x = (w - target_w) // 2
        return int(pos_x), int(pos_y), int(target_w)


@buyer_bp.route("/virtual-tryon", methods=["POST"])
def virtual_tryon():
    """
    Accepts person photo and clothing photo, submits to IDM-VTON diffusion model space
    via gradio_client, saves generated try-on output, and returns result URL.
    """
    # 1. Auth validation (X-User-Email header / form fallback matching app pattern)
    user_email = (
        request.headers.get("X-User-Email")
        or request.headers.get("X-Admin-Email")
        or request.form.get("user_email")
        or request.form.get("email")
    )
    if user_email:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, email FROM users WHERE LOWER(email) = ?", (user_email.strip().lower(),))
            user = cursor.fetchone()
            conn.close()
            if not user:
                return jsonify({"error": "Unauthorized: User not recognized."}), 401
        except Exception as dberr:
            logger.warning(f"Could not verify user email in database: {dberr}")

    # 2. File validation
    if "person" not in request.files or "cloth" not in request.files:
        return jsonify({"error": "Both 'person' and 'cloth' images are required."}), 400

    person_file = request.files["person"]
    cloth_file = request.files["cloth"]

    if not person_file.filename or not cloth_file.filename:
        return jsonify({"error": "Both 'person' and 'cloth' images must be selected."}), 400

    if not allowed_file(person_file.filename) or not allowed_file(cloth_file.filename):
        return jsonify({"error": "Invalid file format. Supported: PNG, JPG, JPEG, WebP."}), 400

    base_id = uuid.uuid4().hex[:10]

    # 3. Save uploaded images
    p_ext = person_file.filename.rsplit(".", 1)[1].lower()
    p_filename = f"person_{base_id}.{p_ext}"
    p_path = os.path.join(PERSON_UPLOAD_FOLDER, p_filename)
    person_file.save(p_path)

    c_ext = cloth_file.filename.rsplit(".", 1)[1].lower()
    c_filename = f"cloth_{base_id}.{c_ext}"
    c_path = os.path.join(CLOTH_UPLOAD_FOLDER, c_filename)
    cloth_file.save(c_path)

    # 4. IDM-VTON Gradio Client Prediction
    space_name = os.getenv("IDM_VTON_SPACE", "yisol/IDM-VTON")
    timeout_sec = int(os.getenv("IDM_VTON_TIMEOUT", 180))
    garment_des = request.form.get("garment_des", "a shirt")

    try:
        from gradio_client import Client
        try:
            from gradio_client import handle_file as gr_file
        except ImportError:
            from gradio_client import file as gr_file

        try:
            client = Client(space_name, httpx_kwargs={"timeout": timeout_sec})
        except Exception as first_err:
            if "CERTIFICATE_VERIFY_FAILED" in str(first_err) or "certificate verify failed" in str(first_err):
                logger.warning("SSL certificate verification failed, retrying with ssl_verify=False...")
                client = Client(space_name, ssl_verify=False, httpx_kwargs={"timeout": timeout_sec})
            else:
                raise first_err
    except Exception as conn_err:
        logger.error(f"Failed to connect to IDM-VTON Space '{space_name}': {conn_err}")
        return jsonify({
            "error": "Virtual try-on service is temporarily unavailable. Could not connect to model space.",
            "details": str(conn_err)
        }), 502

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.predict,
                dict={"background": gr_file(p_path), "layers": [], "composite": None},
                garm_img=gr_file(c_path),
                garment_des=garment_des,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            result = future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        logger.error(f"IDM-VTON prediction timed out after {timeout_sec}s")
        return jsonify({
            "error": "Virtual try-on service is temporarily unavailable. Request timed out."
        }), 502
    except Exception as pred_err:
        logger.error(f"IDM-VTON prediction failed: {pred_err}")
        return jsonify({
            "error": "Virtual try-on service is temporarily unavailable.",
            "details": str(pred_err)
        }), 502

    # 5. Process and save result
    if not result or len(result) == 0:
        return jsonify({
            "error": "Virtual try-on service returned an empty result."
        }), 502

    output_item = result[0]
    output_path = output_item["path"] if (isinstance(output_item, dict) and "path" in output_item) else output_item

    result_filename = f"tryon_{base_id}.jpg"
    result_path = os.path.join(TRYON_OUTPUT_FOLDER, result_filename)

    try:
        with Image.open(output_path) as out_img:
            out_img.convert("RGB").save(result_path, "JPEG", quality=95)
    except Exception as img_err:
        logger.warning(f"PIL conversion failed, copying output directly: {img_err}")
        shutil.copyfile(output_path, result_path)

    tryon_url = f"/uploads/tryon/{result_filename}"

    return jsonify({
        "success": True,
        "tryon_url": tryon_url,
        "result_url": tryon_url,
        "person_url": f"/uploads/persons/{p_filename}",
        "cloth_url": f"/uploads/clothes/{c_filename}",
        "message": "Virtual try-on completed successfully!"
    }), 200


