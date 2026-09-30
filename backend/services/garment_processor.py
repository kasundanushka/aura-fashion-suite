"""
Garment Processor Service
Handles intelligent shirt segmentation, background/annotation removal,
front & back detection, automatic back generation, and 4K Real-Human studio generation.
"""

import os
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'uploads', 'clothes')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_dominant_colors(img, num_colors=3):
    """Extract primary, secondary, and accent colors from the garment."""
    small = img.resize((100, 100)).convert('RGB')
    arr = np.array(small)
    pixels = arr.reshape(-1, 3)
    
    # Bin colors into 16 levels per channel
    binned = (pixels // 32) * 32 + 16
    unique, counts = np.unique(binned, axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    palette = []
    for idx in sorted_indices[:num_colors]:
        c = unique[idx]
        hex_code = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
        palette.append(hex_code)
    
    return palette


def isolate_garment_background(img):
    """
    Cleans and isolates garment from mockup/photo background using gradient-aware
    floodfill, morphological filtering, and largest connected component extraction.
    Preserves inner fabric, colors, and graphics with pristine edge fidelity.
    """
    arr = np.array(img.convert('RGB')).astype(float)
    h, w, _ = arr.shape
    
    gray = np.mean(arr, axis=2)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_blur = ndimage.gaussian_filter(np.hypot(gx, gy), 1.5)
    
    blurred = ndimage.gaussian_filter(arr, (2.0, 2.0, 0))
    corner_samples = [
        blurred[:25, :25],
        blurred[:25, -25:],
        blurred[-25:, :25],
        blurred[-25:, -25:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corner_samples], axis=0)
    diff_to_bg = np.sqrt(np.sum((blurred - bg_color)**2, axis=2))
    
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        visited[0, x] = True; q.append((0, x))
        visited[h - 1, x] = True; q.append((h - 1, x))
    for y in range(h):
        visited[y, 0] = True; q.append((y, 0))
        visited[y, w - 1] = True; q.append((y, w - 1))
        
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                pix = arr[ny, nx]
                sat = np.max(pix) - np.min(pix)
                is_garment = (grad_blur[ny, nx] > 85 and diff_to_bg[ny, nx] > 25) or (sat > 25) or (diff_to_bg[ny, nx] > 45)
                if not is_garment:
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    
    mask = ~visited
    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_opening(mask, structure=struct, iterations=2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    
    labels, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = (labels == (np.argmax(sizes) + 1))
        
    mask = ndimage.binary_fill_holes(mask)
    feather = ndimage.gaussian_filter(mask.astype(float), sigma=1.2)
    rgba = np.dstack([arr, feather * 255.0]).astype(np.uint8)
    return Image.fromarray(rgba)


def blend_garment_onto_base(base_img, clean_garment_rgba, box, target_lum_ref=175.0):
    """
    Blends a clean garment RGBA texture onto a high-res studio model image.
    Uses luminosity matching for realistic fabric folds, shadows, and natural 3D contours.
    """
    base_np = np.array(base_img).astype(float)
    min_x, min_y, box_w, box_h = box
    
    # Tight crop clean garment to active bounds first
    alpha_raw = np.array(clean_garment_rgba)[:, :, 3]
    ys, xs = np.where(alpha_raw > 10)
    if len(ys) > 0 and len(xs) > 0:
        cropped_garment = clean_garment_rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    else:
        cropped_garment = clean_garment_rgba
        
    g_resized = cropped_garment.resize((box_w, box_h), Image.Resampling.LANCZOS)
    g_np = np.array(g_resized).astype(float)
    
    alpha = (g_np[:, :, 3] / 255.0)[:, :, np.newaxis]
    base_crop = base_np[min_y:min_y+box_h, min_x:min_x+box_w]
    base_lum = np.mean(base_crop, axis=2)
    shade_factor = np.clip(base_lum / target_lum_ref, 0.72, 1.25)[:, :, np.newaxis]
    
    cloth_rgb = g_np[:, :, :3] * shade_factor
    blended = base_crop * (1.0 - alpha) + cloth_rgb * alpha
    base_np[min_y:min_y+box_h, min_x:min_x+box_w] = np.clip(blended, 0, 255)
    return Image.fromarray(base_np.astype(np.uint8))


def classify_garment(clean_rgba):
    """
    Intelligently classifies a segmented clothing image into:
    - 'bottom': Trousers / Pants / Jeans / Shorts
    - 'top': Shirt / T-Shirt / Polo / Hoodie / Jacket
    - 'dress': Dress / Gown
    Uses geometric aspect ratio, crotch bifurcation (two legs vs single torso),
    and contour width variation.
    """
    arr = np.array(clean_rgba)
    alpha = arr[:, :, 3] > 25
    ys, xs = np.where(alpha)
    if len(ys) == 0 or len(xs) == 0:
        return 'top', 'Shirt / Top', '👕', 'ෂර්ට් / Top', 'Upper Torso & Shoulders'
        
    crop_mask = alpha[ys.min():ys.max()+1, xs.min():xs.max()+1]
    gh, gw = crop_mask.shape
    if gh < 20 or gw < 20:
        return 'top', 'Shirt / Top', '👕', 'ෂර්ට් / Top', 'Upper Torso & Shoulders'
        
    aspect = gh / float(gw)
    
    # Analyze row components in lower section (55% to 85% of garment height)
    lower_start = int(gh * 0.55)
    lower_end = int(gh * 0.85)
    sampled_rows = 0
    two_legs_count = 0
    
    step = max(1, (lower_end - lower_start) // 25)
    for y in range(lower_start, lower_end, step):
        sampled_rows += 1
        row = crop_mask[y, :]
        diff = np.diff(row.astype(int))
        rising = np.sum(diff == 1)
        falling = np.sum(diff == -1)
        
        mid_third = row[int(gw * 0.35):int(gw * 0.65)]
        has_center_gap = np.any(~mid_third) and np.any(row[:int(gw * 0.35)]) and np.any(row[int(gw * 0.65):])
        
        if rising >= 2 and falling >= 2 and has_center_gap:
            two_legs_count += 1
            
    split_ratio = two_legs_count / float(max(1, sampled_rows))
    
    # If there is a leg split across the lower section, it is pants/trousers
    if split_ratio >= 0.25 or (aspect >= 1.30 and split_ratio >= 0.15):
        return 'bottom', 'Trousers / Pants', '👖', 'කලිසම / Pants', 'Lower Body (Waist & Legs)'
    elif aspect >= 1.38 and split_ratio < 0.10:
        top_w = np.sum(crop_mask[int(gh * 0.25), :])
        bot_w = np.sum(crop_mask[int(gh * 0.82), :])
        if bot_w > top_w * 1.15:
            return 'dress', 'Dress / Gown', '👗', 'ගවුම / Dress', 'Full Body (Chest to Knees)'
        return 'top', 'Shirt / Top', '👕', 'ෂර්ට් / Top', 'Upper Torso & Shoulders'
    else:
        return 'top', 'Shirt / Top', '👕', 'ෂර්ට් / Top', 'Upper Torso & Shoulders'


def generate_4k_studio_renders(front_img, back_img, filename_base, category='top'):
    """
    Generates 4K Real-Human studio try-on renders for Male, Female, and Back views.
    Intelligently dresses the garment on the correct anatomical area based on category:
    - 'top': Fitted onto upper chest/torso
    - 'bottom': Fitted onto waist and legs (trousers/jeans)
    - 'dress': Fitted continuously from chest down past knees
    """
    male_base_path = os.path.join(UPLOAD_DIR, 'base_model_male.jpg')
    female_base_path = os.path.join(UPLOAD_DIR, 'base_model_female.jpg')
    back_base_path = os.path.join(UPLOAD_DIR, 'base_model_back.jpg')
    
    clean_front = isolate_garment_background(front_img)
    clean_back = isolate_garment_background(back_img)
    
    male_url = '/uploads/clothes/model_male_front.jpg'
    female_url = '/uploads/clothes/model_female_front.jpg'
    back_url = '/uploads/clothes/model_back.jpg'
    
    # Anatomical bounding boxes per category
    if category == 'bottom':
        # Lower body boxes: from waistband down to shoes
        male_box = (270, 595, 360, 535)
        female_box = (310, 590, 280, 525)
        back_box = (265, 600, 365, 535)
        male_lum = female_lum = back_lum = 155.0
    elif category == 'dress':
        # Full body dress boxes
        male_box = (260, 230, 380, 720)
        female_box = (315, 292, 266, 680)
        back_box = (255, 220, 385, 720)
        male_lum, female_lum, back_lum = 175.0, 165.0, 160.0
    else:
        # Upper torso shirt boxes
        male_box = (260, 230, 380, 402)
        female_box = (315, 292, 266, 328)
        back_box = (255, 220, 385, 415)
        male_lum, female_lum, back_lum = 180.0, 165.0, 160.0
    
    try:
        if os.path.exists(male_base_path):
            male_base = Image.open(male_base_path).convert('RGB')
            male_fitted = blend_garment_onto_base(male_base, clean_front, male_box, target_lum_ref=male_lum)
            male_filename = f"{filename_base}_model_male.jpg"
            male_fitted.save(os.path.join(UPLOAD_DIR, male_filename), 'JPEG', quality=95)
            male_url = f"/uploads/clothes/{male_filename}"
            
        if os.path.exists(female_base_path):
            female_base = Image.open(female_base_path).convert('RGB')
            female_fitted = blend_garment_onto_base(female_base, clean_front, female_box, target_lum_ref=female_lum)
            female_filename = f"{filename_base}_model_female.jpg"
            female_fitted.save(os.path.join(UPLOAD_DIR, female_filename), 'JPEG', quality=95)
            female_url = f"/uploads/clothes/{female_filename}"
            
        if os.path.exists(back_base_path):
            back_base = Image.open(back_base_path).convert('RGB')
            back_fitted = blend_garment_onto_base(back_base, clean_back, back_box, target_lum_ref=back_lum)
            back_filename = f"{filename_base}_model_back.jpg"
            back_fitted.save(os.path.join(UPLOAD_DIR, back_filename), 'JPEG', quality=95)
            back_url = f"/uploads/clothes/{back_filename}"
    except Exception as err:
        print(f"Error in generate_4k_studio_renders: {err}")
        
    return male_url, female_url, back_url


def process_garment_image(image_path, filename_base='processed'):
    """
    Analyzes an uploaded clothing image:
    1. Identifies if it's a pair of pants/trousers, a single shirt, or dual-view shirt mockup.
    2. Crops and isolates the front item with clean background extraction.
    3. Detects category ('bottom' vs 'top' vs 'dress').
    4. Crops or auto-synthesizes the matching back view.
    5. Extracts dominant color palette.
    6. Synthesizes 4K Real-Human studio renders on male, female, and back models in correct anatomical position.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    aspect = h / float(w)
    
    # Sample background color from corners
    arr = np.array(img).astype(float)
    corner_samples = [
        arr[5:35, 5:35],
        arr[5:35, -35:-5],
        arr[-35:-5, 5:35],
        arr[-35:-5, -35:-5]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corner_samples], axis=0)
    
    # Check if the entire image is a pair of pants (tall aspect + dual legs split on isolated mask)
    is_pants = False
    clean_full_garment = None
    if aspect >= 1.15:
        clean_full_garment = isolate_garment_background(img)
        arr_f = np.array(clean_full_garment)[:, :, 3] > 25
        ys_f, xs_f = np.where(arr_f)
        if len(ys_f) > 0 and len(xs_f) > 0:
            crop_f = arr_f[ys_f.min():ys_f.max()+1, xs_f.min():xs_f.max()+1]
            gh_f, gw_f = crop_f.shape
            l_start = int(gh_f * 0.55)
            l_end = int(gh_f * 0.85)
            two_legs = 0
            total_rows = 0
            for y in range(l_start, l_end, max(1, (l_end - l_start) // 20)):
                total_rows += 1
                row = crop_f[y, :]
                diff_r = np.diff(row.astype(int))
                rising = np.sum(diff_r == 1)
                falling = np.sum(diff_r == -1)
                mid_third = row[int(gw_f * 0.35):int(gw_f * 0.65)]
                has_center_gap = np.any(~mid_third) and np.any(row[:int(gw_f * 0.35)]) and np.any(row[int(gw_f * 0.65):])
                if rising >= 2 and falling >= 2 and has_center_gap:
                    two_legs += 1
            split_ratio = two_legs / float(max(1, total_rows))
            if split_ratio >= 0.55:
                is_pants = True
            
    # Check if image has dual views stacked vertically (aspect ratio >= 1.08 and not pants)
    is_dual = (aspect >= 1.08) and (not is_pants)
    
    front_img = None
    back_img = None
    has_back = False
    auto_back = False
    
    if is_dual:
        # Dual-View Mockup:
        # Top half contains FRONT, bottom half contains BACK.
        left_bound = int(w * 0.16)
        right_bound = int(w * 0.84)
        
        front_top = int(h * 0.015)
        front_bottom = int(h * 0.495)
        front_crop = img.crop((left_bound, front_top, right_bound, front_bottom))
        
        back_top = int(h * 0.505)
        back_bottom = int(h * 0.985)
        back_crop = img.crop((left_bound, back_top, right_bound, back_bottom))
        
        # Verify if back half contains garment pixels
        back_arr = np.array(back_crop).astype(float)
        back_diff = np.sqrt(np.sum((back_arr - bg_color)**2, axis=2))
        if np.mean(back_diff > 30) > 0.12:
            has_back = True
            front_img = front_crop
            back_img = back_crop
        else:
            front_img = front_crop
            has_back = False
    else:
        # Single-View
        left_bound = int(w * 0.08)
        right_bound = int(w * 0.92)
        front_crop = img.crop((left_bound, int(h * 0.02), right_bound, int(h * 0.98)))
        front_img = front_crop
        has_back = False
    
    # Isolate garment
    if is_pants and clean_full_garment is not None:
        clean_front = clean_full_garment
    else:
        clean_front = isolate_garment_background(front_img)
    
    # Intelligent Garment Classification
    category, category_name, category_icon, category_label_si, target_area = classify_garment(clean_front)
    
    palette = extract_dominant_colors(front_img, 3)
    primary_color = palette[0] if palette else '#ffffff'
    secondary_color = palette[1] if len(palette) > 1 else '#1a3a6b'
    
    # If no back view detected, auto-generate matching back based on garment category
    if not has_back:
        auto_back = True
        back_w, back_h = front_img.size
        back_img = generate_matching_back(back_w, back_h, primary_color, secondary_color, category=category)
    
    # Save processed textures
    front_filename = f"{filename_base}_front.png"
    back_filename = f"{filename_base}_back.png"
    
    front_save_path = os.path.join(UPLOAD_DIR, front_filename)
    back_save_path = os.path.join(UPLOAD_DIR, back_filename)
    
    clean_front.save(front_save_path, 'PNG')
    if has_back:
        clean_back = isolate_garment_background(back_img)
        clean_back.save(back_save_path, 'PNG')
    else:
        back_img.save(back_save_path, 'PNG')
    
    # Generate 4K Real-Human Studio Renders with category-specific placement
    male_front_url, female_front_url, model_back_url = generate_4k_studio_renders(
        clean_front, back_img, filename_base, category=category
    )
    
    return {
        'front_url': f"/uploads/clothes/{front_filename}",
        'back_url': f"/uploads/clothes/{back_filename}",
        'has_back': has_back,
        'auto_generated_back': auto_back,
        'palette': palette,
        'primary_color': primary_color,
        'secondary_color': secondary_color,
        'category': category,
        'category_name': category_name,
        'category_icon': category_icon,
        'category_label_si': category_label_si,
        'target_area': target_area,
        'model_male_front': male_front_url,
        'model_female_front': female_front_url,
        'model_back': model_back_url
    }


def generate_matching_back(w, h, primary_hex, secondary_hex, category='top'):
    """
    Auto-synthesizes a clean matching back view tailored to garment type:
    - 'bottom': Back of trousers with back waistband, yoke, and twin pockets.
    - 'top' / 'dress': Back of shirt with collar, yoke line, and side panels.
    """
    def hex_to_rgb(hx):
        hx = hx.lstrip('#')
        if len(hx) == 3:
            hx = ''.join([c*2 for c in hx])
        return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))
    
    prim_rgb = hex_to_rgb(primary_hex)
    sec_rgb = hex_to_rgb(secondary_hex)
    
    back = Image.new('RGB', (w, h), prim_rgb)
    draw = ImageDraw.Draw(back)
    
    if category == 'bottom':
        # Back waistband
        wb_h = max(4, int(h * 0.08))
        draw.rectangle([0, 0, w, wb_h], fill=sec_rgb)
        
        # Back yoke line (V-shape)
        yoke_top = wb_h
        yoke_mid = int(h * 0.16)
        draw.polygon([(0, yoke_top), (w // 2, yoke_mid), (w, yoke_top)], fill=sec_rgb)
        
        # Dual back pockets
        pocket_w = int(w * 0.28)
        pocket_h = int(h * 0.18)
        pocket_y = int(h * 0.20)
        
        # Left pocket
        l_x1 = int(w * 0.15)
        draw.rectangle([l_x1, pocket_y, l_x1 + pocket_w, pocket_y + pocket_h], outline=sec_rgb, width=max(2, int(w * 0.01)))
        # Right pocket
        r_x1 = int(w * 0.57)
        draw.rectangle([r_x1, pocket_y, r_x1 + pocket_w, pocket_y + pocket_h], outline=sec_rgb, width=max(2, int(w * 0.01)))
    else:
        # Back collar band
        collar_h = int(h * 0.12)
        collar_w = int(w * 0.40)
        cx = w // 2
        draw.chord(
            [cx - collar_w // 2, -collar_h // 2, cx + collar_w // 2, collar_h],
            start=0, end=180, fill=sec_rgb
        )
        
        # Back yoke line
        draw.line([(int(w * 0.2), int(h * 0.18)), (int(w * 0.8), int(h * 0.18))], fill=sec_rgb, width=max(2, int(w * 0.01)))
        
        # Side curved accent panels matching front styling
        side_w = int(w * 0.14)
        # Left side panel
        draw.polygon([
            (0, int(h * 0.35)),
            (side_w, int(h * 0.70)),
            (0, int(h * 0.95))
        ], fill=sec_rgb)
        
        # Right side panel
        draw.polygon([
            (w, int(h * 0.35)),
            (w - side_w, int(h * 0.70)),
            (w, int(h * 0.95))
        ], fill=sec_rgb)
        
    back = back.filter(ImageFilter.SMOOTH_MORE)
    return back
