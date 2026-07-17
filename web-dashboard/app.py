# -*- coding: utf-8 -*-
"""
Flask Face Parsing Dashboard
Serves the face analysis web dashboard using:
 - SegFormer (jonathandinu/face-parsing) loaded locally from ./face-parsing/
 - MediaPipe Face Mesh for 468 landmark-based metrics
"""

import os
import io
import base64
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import mediapipe as mp
from sklearn.cluster import KMeans
from collections import Counter
from inference_sdk import InferenceHTTPClient
import math
from PIL import ImageDraw
import tempfile
from scipy.interpolate import splprep, splev
from scipy.stats import skew as scipy_skew
# App setup
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face-parsing")

# ─────────────────────────────────────────────
# Label map (19 classes from model config)
# ─────────────────────────────────────────────
LABEL_MAP = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_g",
    4: "l_eye",
    5: "r_eye",
    6: "l_brow",
    7: "r_brow",
    8: "l_ear",
    9: "r_ear",
    10: "mouth",
    11: "u_lip",
    12: "l_lip",
    13: "hair",
    14: "hat",
    15: "ear_r",
    16: "neck_l",
    17: "neck",
    18: "cloth",
}

# ─────────────────────────────────────────────
# Global model cache (loaded once at startup)
# ─────────────────────────────────────────────
_segformer_processor = None
_segformer_model = None
_face_mesh = None
_roboflow_client = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    global _segformer_processor, _segformer_model, _face_mesh, _roboflow_client
    if _segformer_processor is None:
        print(f"[INFO] Loading SegFormer from {MODEL_PATH} on {_device}...")
        _segformer_processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        _segformer_model.to(_device)
        _segformer_model.eval()
        print("[INFO] SegFormer loaded.")
    if _face_mesh is None:
        print("[INFO] Loading MediaPipe Face Mesh...")
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        print("[INFO] MediaPipe loaded.")
    if _roboflow_client is None:
        _roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.environ.get("ROBOFLOW_API_KEY", "")
        )
        print("[INFO] Roboflow client initialized.")


# ─────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────

def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_b64(arr: np.ndarray) -> str:
    """Convert RGB numpy array to base64 JPEG data URI, resizing if too large to prevent QuotaExceededError."""
    if arr is None: return None
    img_pil = Image.fromarray(arr.astype(np.uint8))
    if max(img_pil.size) > 800:
        img_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _get_mask(labels, label_idx):
    if isinstance(label_idx, (list, tuple)):
        return np.isin(labels, label_idx)
    return labels == label_idx

def extract_part_white_bg(img_rgb: np.ndarray, labels: np.ndarray, label_idx) -> np.ndarray:
    """Return full-size image with only the target part visible on white background."""
    mask = _get_mask(labels, label_idx)
    part = np.ones_like(img_rgb) * 255
    part[mask] = img_rgb[mask]
    return part.astype(np.uint8)


def extract_part_cropped_raw(img_rgb: np.ndarray, labels: np.ndarray, label_idx):
    """
    Return a tight bounding-box crop of the ORIGINAL image pixels
    (no masking — identical to the notebook: crop = img[y1:y2+1, x1:x2+1]).
    """
    mask = _get_mask(labels, label_idx)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return img_rgb[y1:y2+1, x1:x2+1].astype(np.uint8)  # raw crop, no mask applied


def extract_part_cropped_whitebg(img_rgb: np.ndarray, labels: np.ndarray, label_idx):
    """
    Return a tight bounding-box crop where ONLY the segmented part is visible
    and the rest of the crop is white (masked crop).
    """
    mask = _get_mask(labels, label_idx)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    white = np.ones_like(img_rgb) * 255
    white[mask] = img_rgb[mask]
    return white[y1:y2+1, x1:x2+1].astype(np.uint8)  # masked crop on white


def part_images_b64(img_rgb: np.ndarray, labels: np.ndarray, label_idx):
    """
    Returns (white_bg_b64, raw_crop_b64) where:
      - white_bg  = tightly cropped image, part visible, rest white
      - raw_crop  = tight bbox crop of original pixels (notebook-style)
    """
    white_bg  = extract_part_cropped_whitebg(img_rgb, labels, label_idx)
    raw_crop  = extract_part_cropped_raw(img_rgb, labels, label_idx)
    white_bg_b64  = rgb_to_b64(white_bg) if white_bg is not None else None
    raw_crop_b64  = rgb_to_b64(raw_crop) if raw_crop is not None else None
    return white_bg_b64, raw_crop_b64


def extract_chin_mediapipe(img_rgb: np.ndarray, pts: np.ndarray):
    """
    Extracts the chin region using MediaPipe landmarks and smooth B-spline interpolation.
    Returns (white_crop, raw_crop) similar to part_images_b64 output.
    """
    if pts is None or len(pts) == 0:
        return None, None
        
    CHIN_INDICES = [
        204, 83, 18, 313, 424, 431,
        395, 369, 396, 175, 171,
        140, 170, 211
    ]
    
    # Create smooth polygon using B-spline interpolation
    pts_arr = np.array([pts[idx] for idx in CHIN_INDICES])
    pts_arr = np.vstack((pts_arr, pts_arr[0])) # Close polygon
    
    tck, u = splprep([pts_arr[:, 0], pts_arr[:, 1]], s=0, per=True)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)
    chin_poly = np.int32(np.vstack((out[0], out[1])).T)
    
    # Apply chin overlay
    overlay = img_rgb.copy()
    CHIN_COLOR = (170, 185, 185)  # RGB pastel color
    cv2.fillPoly(overlay, [chin_poly], CHIN_COLOR)
    
    # Blend overlay with original image
    alpha = 0.65
    final_image = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    return final_image, final_image


def extract_cheeks_mediapipe(img_rgb: np.ndarray, pts: np.ndarray):
    """
    Extracts the cheeks region using MediaPipe landmarks and smooth B-spline interpolation.
    Returns (white_crop, raw_crop) which are actually the full image with cheeks highlighted.
    """
    if pts is None or len(pts) == 0:
        return None, None

    RIGHT_CHEEK_INDICES = [114, 120, 47, 142, 203, 205, 207, 213, 215, 138, 132, 177, 147, 137, 234, 227, 116, 117, 118, 119, 121]
    LEFT_CHEEK_INDICES = [343, 349, 277, 371, 423, 425, 427, 433, 435, 367, 361, 401, 376, 366, 454, 447, 345, 346, 347, 348, 350]

    def get_smooth_polygon(indices):
        pts_arr = np.array([pts[idx] for idx in indices])
        pts_arr = np.vstack((pts_arr, pts_arr[0])) # Close polygon
        
        tck, u = splprep([pts_arr[:, 0], pts_arr[:, 1]], s=0, per=True)
        unew = np.linspace(0, 1, 100)
        out = splev(unew, tck)
        smooth_pts = np.int32(np.vstack((out[0], out[1])).T)
        return smooth_pts

    right_pts = get_smooth_polygon(RIGHT_CHEEK_INDICES)
    left_pts = get_smooth_polygon(LEFT_CHEEK_INDICES)

    overlay = img_rgb.copy()
    CHEEK_COLOR = (150, 170, 180)  # RGB color for cheek highlight
    cv2.fillPoly(overlay, [right_pts], CHEEK_COLOR)
    cv2.fillPoly(overlay, [left_pts], CHEEK_COLOR)

    alpha = 0.65
    final_image = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    return final_image, final_image


def extract_jaw_mediapipe(pil_img: Image.Image):
    """
    Extracts the jaw area from a 45-degree side profile image using MediaPipe face landmarks.
    Draws 478 landmarks and applies a dynamic polygon mask over the jaw.
    Returns: cropped_b64, overlay_b64
    """
    img_rgb = np.array(pil_img.convert('RGB'))
    h, w, _ = img_rgb.shape
    
    # Process the image to get landmarks with z-coordinates
    results = _face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, None
        
    face_landmarks = results.multi_face_landmarks[0].landmark
    
    # Draw all 478 landmarks (beige color)
    annotated_image = img_rgb.copy()
    for landmark in face_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(annotated_image, (x, y), 1, (237, 234, 222), -1)
        
    # Determine face orientation using 3D Z-coordinates
    left_ear_z = face_landmarks[454].z
    right_ear_z = face_landmarks[234].z
    facing_left = right_ear_z < left_ear_z  # Right ear is closer to camera
    
    if facing_left:
        jawline_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
        ear_idx = 234
    else:
        jawline_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
        ear_idx = 454
        
    nose_tip = face_landmarks[1]
    ear = face_landmarks[ear_idx]
    
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    ear_x, ear_y = int(ear.x * w), int(ear.y * h)
    
    # Calculate the front boundary (65% distance)
    if facing_left:
        front_x = int(ear_x + 0.65 * (nose_x - ear_x))
    else:
        front_x = int(ear_x - 0.65 * (ear_x - nose_x))
        
    polygon_pts = []
    # 1. Start at top-back corner
    polygon_pts.append([ear_x, nose_y])
    # 2. Add top-front corner
    polygon_pts.append([front_x, nose_y])
    
    # 3. Trace the jawline backwards
    jawline_pts = []
    for idx in jawline_indices:
        pt_x = int(face_landmarks[idx].x * w)
        pt_y = int(face_landmarks[idx].y * h)
        # Push down for beard clearance (3% of height)
        pt_y += int(h * 0.03)
        # Only include points behind the front cutoff line
        if (facing_left and pt_x <= front_x) or (not facing_left and pt_x >= front_x):
            jawline_pts.append([pt_x, pt_y])
            
    jawline_pts = jawline_pts[::-1]
    polygon_pts.extend(jawline_pts)
    
    # 4. Close the polygon at the ear
    polygon_pts.append([ear_x, ear_y])
    
    polygon_pts = np.array(polygon_pts, np.int32)
    
    # Create mask and apply overlay to ORIGINAL image (no dots)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon_pts], 1)
    
    overlay = img_rgb.copy()
    overlay[poly_mask == 1] = [170, 170, 170]
    
    alpha = 0.55
    highlight_image = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    dots_b64 = rgb_to_b64(annotated_image)
    highlight_b64 = rgb_to_b64(highlight_image)
    
    return dots_b64, highlight_b64


# ─────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────

def run_segmentation(pil_img: Image.Image) -> np.ndarray:
    """Run SegFormer and return pixel-wise label map (H x W)."""
    inputs = _segformer_processor(images=pil_img, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _segformer_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()
    return labels


# ─────────────────────────────────────────────
# Landmark detection
# ─────────────────────────────────────────────

def run_landmarks(img_rgb: np.ndarray):
    """Returns (landmark_points_array, success_bool)."""
    results = _face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None, False
    landmarks = results.multi_face_landmarks[0]
    h, w = img_rgb.shape[:2]
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts), True


# ─────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────

def lm(pts, idx):
    return pts[idx]

def dist(pts, i, j):
    return float(np.linalg.norm(pts[j] - pts[i]))

def angle_deg(pts, vertex, a, b):
    v, pa, pb = pts[vertex], pts[a], pts[b]
    va, vb = pa - v, pb - v
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def angle_from_vertical(pts, top, bottom):
    vec = pts[bottom] - pts[top]
    cos_a = np.dot(vec, [0, 1]) / (np.linalg.norm(vec) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def angle_from_horizontal(pts, a, b):
    vec = pts[b] - pts[a]
    cos_a = np.dot(vec, [1, 0]) / (np.linalg.norm(vec) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def curvature_ratio(pts, start, mid, end):
    p1, p2, p3 = pts[start], pts[mid], pts[end]
    chord = p3 - p1
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-6:
        return 0.0
    t = np.dot(p2 - p1, chord) / (chord_len ** 2)
    proj = p1 + t * chord
    sagitta = np.linalg.norm(p2 - proj)
    return float(sagitta / chord_len)

def eye_aspect_ratio(pts, p1, p2, p3, p4, p5, p6):
    pp1, pp2, pp3, pp4, pp5, pp6 = [pts[i] for i in (p1, p2, p3, p4, p5, p6)]
    vertical = np.linalg.norm(pp2 - pp6) + np.linalg.norm(pp3 - pp5)
    horizontal = np.linalg.norm(pp1 - pp4)
    return float(vertical / (2 * horizontal + 1e-9))

def point_line_deviation(pts, point_idx, line_i1, line_i2):
    p, a, b = pts[point_idx], pts[line_i1], pts[line_i2]
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


# ─────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────

def _fmt(val, decimals=2):
    """Format a float or return 'N/A'."""
    if val is None:
        return "N/A"
    return f"{round(float(val), decimals)}"

def classify_ear(ear):
    if ear is None: return "N/A"
    if ear < 0.25: return "Narrow / Closed"
    if ear < 0.30: return "Slightly Narrow"
    if ear < 0.36: return "Average"
    if ear < 0.40: return "Wide Open"
    return "Very Wide"

def classify_brow_position(height_mm):
    if height_mm is None: return "N/A"
    if height_mm < 14: return "Low-set"
    if height_mm < 18: return "Average"
    return "High-set"

def classify_brow_tilt(apex_angle):
    if apex_angle is None: return "N/A"
    if apex_angle < 130: return "Strongly Arched"
    if apex_angle < 140: return "Gently Arched"
    if apex_angle < 150: return "Flat / Horizontal"
    return "Slightly Downward"

def classify_brow_shape(curvature):
    if curvature is None: return "N/A"
    if curvature < 0.05: return "Straight"
    if curvature < 0.12: return "Gently Arched"
    return "Strongly Arched"

def classify_brow_virility(elevation_ratio):
    if elevation_ratio is None: return "N/A"
    if elevation_ratio < 0.15: return "Delicate"
    if elevation_ratio < 0.25: return "Average"
    return "Bold / Prominent"

def classify_nose_shape(aspect_ratio):
    if aspect_ratio is None: return "N/A"
    if aspect_ratio < 0.6: return "Narrow / Leptorrhine"
    if aspect_ratio < 0.85: return "Average / Mesorrhine"
    return "Wide / Platyrrhine"

def classify_nose_tip(naso_canthal):
    if naso_canthal is None: return "N/A"
    if naso_canthal < 0.85: return "Narrow / Refined"
    if naso_canthal < 1.05: return "Average"
    return "Wide / Bulbous"

def classify_nose_height(height_mm):
    if height_mm is None: return "N/A"
    if height_mm < 40: return "Short"
    if height_mm < 55: return "Average"
    return "Tall"

def classify_nose_width(width_mm):
    if width_mm is None: return "N/A"
    if width_mm < 30: return "Narrow"
    if width_mm < 40: return "Average"
    return "Wide"

def classify_lip_fullness(mouth_width, philtrum):
    if mouth_width is None: return "N/A"
    if mouth_width < 40: return "Thin / Narrow"
    if mouth_width < 55: return "Average"
    return "Full / Wide"

def classify_lip_proportions(cupid_angle):
    if cupid_angle is None: return "N/A"
    if cupid_angle < 100: return "Peaked / Defined"
    if cupid_angle < 130: return "Balanced"
    return "Flat / Subtle"

def classify_cheek_width(malar_ratio):
    if malar_ratio is None: return "N/A"
    if malar_ratio < 0.75: return "Narrow"
    if malar_ratio < 0.90: return "Average"
    return "Wide"

def classify_cheek_position(pos_ratio):
    if pos_ratio is None: return "N/A"
    if pos_ratio < 0.40: return "High-set"
    if pos_ratio < 0.55: return "Average"
    return "Low-set"

def classify_cheek_fullness(malar_ratio):
    if malar_ratio is None: return "N/A"
    if malar_ratio < 0.75: return "Flat / Lean"
    if malar_ratio < 0.90: return "Average"
    return "Full / Prominent"

def classify_eye_tilt(pts):
    """Positive tilt = outer corner higher than inner (y is smaller in image coords)."""
    try:
        r_inner_y = pts[133][1]
        r_outer_y = pts[33][1]
        l_inner_y = pts[362][1]
        l_outer_y = pts[263][1]
        r_tilt = r_inner_y - r_outer_y
        l_tilt = l_inner_y - l_outer_y
        avg = (r_tilt + l_tilt) / 2
        if avg > 4: return "Positive (Hunter)"
        if avg < -4: return "Negative (Downturned)"
        return "Neutral"
    except Exception:
        return "N/A"

def classify_eyelid_exposure(ear):
    if ear is None: return "N/A"
    if ear < 0.27: return "Low Exposure"
    if ear < 0.34: return "Average"
    return "High Exposure"

def classify_sclera():
    return "N/A"  # Cannot determine from geometry alone

def classify_under_eye():
    return "N/A"  # Cannot determine from geometry alone


# Jaw classifiers
def classify_jaw_definition(jaw_rise_mm):
    if jaw_rise_mm is None: return "N/A"
    if jaw_rise_mm < 40: return "Soft / Rounded"
    if jaw_rise_mm < 60: return "Mild Definition"
    return "Strong / Angular"

def classify_jaw_width(jaw_width_mm):
    if jaw_width_mm is None: return "N/A"
    if jaw_width_mm < 100: return "Narrow"
    if jaw_width_mm < 125: return "Standard"
    return "Wide"

def classify_jaw_shape(r_angle, l_angle):
    if r_angle is None or l_angle is None: return "N/A"
    avg = (r_angle + l_angle) / 2
    if avg < 40: return "V-Shaped"
    if avg < 60: return "U-Shaped"
    return "Square"

def classify_jaw_angle(r_angle, l_angle):
    if r_angle is None or l_angle is None: return "N/A"
    return f"{round((r_angle + l_angle) / 2)}"


# Chin classifiers
def classify_chin_width(chin_w_mm):
    if chin_w_mm is None: return "N/A"
    if chin_w_mm < 35: return "Narrow"
    if chin_w_mm < 50: return "Standard"
    return "Wide"

def classify_chin_projection(dev_mm):
    if dev_mm is None: return "N/A"
    if dev_mm < 3: return "Neutral"
    if dev_mm < 7: return "Slight Deviation"
    return "Deviated"

def classify_chin_shape(chin_w_mm, chin_h_mm):
    if chin_w_mm is None or chin_h_mm is None: return "N/A"
    ratio = chin_w_mm / (chin_h_mm + 1e-9)
    if ratio < 1.5: return "Pointed"
    if ratio < 2.2: return "Round"
    return "Square"

def classify_chin_depth(chin_h_mm):
    if chin_h_mm is None: return "N/A"
    if chin_h_mm < 12: return "Short"
    if chin_h_mm < 20: return "Average"
    return "Deep"


# Hair / Forehead classifiers
def classify_temple_width(fw_mm):
    if fw_mm is None: return "N/A"
    if fw_mm < 95: return "Narrow"
    if fw_mm < 115: return "Average"
    return "Broad"

def classify_hair_volume():
    return "N/A"  # cannot be measured from landmarks

def classify_hair_density():
    return "N/A"  # cannot be measured from landmarks

def classify_hairline_shape():
    return "N/A"  # cannot be measured from landmarks


# ─────────────────────────────────────────────
# Hair Color Analysis
# ─────────────────────────────────────────────
def classify_hair_color(hex_color):
    """Classify hair color into categories based on dominant hex."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    brightness = (r + g + b) / 3
    if r > g and r > b:
        if brightness > 200: return "Light / Golden Blonde"
        if brightness > 150: return "Dark Blonde / Honey"
        if brightness > 100: return "Light Brown / Chestnut"
        if brightness > 50:  return "Dark Brown"
        return "Black / Very Dark Brown"
    if g > r and g > b:
        return "Green / Dyed Hair"
    if b > r and b > g:
        return "Blue / Dyed Hair"
    # neutral
    if brightness > 200: return "Platinum / Light Blonde"
    if brightness > 150: return "Light Brown / Ash Blonde"
    if brightness > 100: return "Medium Brown"
    if brightness > 50:  return "Dark Brown"
    return "Black / Very Dark Brown"


def analyze_hair_color(img_rgb, hair_mask, n_colors=5):
    """
    Given the full RGB image and a boolean hair_mask, extract dominant
    hair colors via KMeans.  Returns a dict with:
      - primary_category  : str  (e.g. 'Dark Brown')
      - primary_hex       : str  (e.g. '#503F37')
      - palette           : list of {hex, rgb, percentage, category}
    """
    try:
        hair_pixels = img_rgb[hair_mask]
        if len(hair_pixels) < n_colors:
            return {"primary_category": "N/A", "primary_hex": None, "palette": []}

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(hair_pixels)
        centers = kmeans.cluster_centers_.astype(int)
        label_counts = Counter(kmeans.labels_)
        total = len(kmeans.labels_)
        sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        palette = []
        for color_idx, count in sorted_colors:
            rgb = centers[color_idx]
            hex_code = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            palette.append({
                "hex": hex_code,
                "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                "percentage": round((count / total) * 100, 1),
                "category": classify_hair_color(hex_code),
            })

        primary = palette[0]
        return {
            "primary_category": primary["category"],
            "primary_hex": primary["hex"],
            "palette": palette,
        }
    except Exception as e:
        print(f"[WARN] Hair color analysis failed: {e}")
        return {"primary_category": "N/A", "primary_hex": None, "palette": []}



# ─────────────────────────────────────────────
# Skin Analysis
# ─────────────────────────────────────────────

# MediaPipe landmark indices used by the notebook
_FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
_LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_EYEBROW = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_EYEBROW = [300,293,334,296,336,285,295,282,283,276]
_L_UNDER_EYE = [110,205,50,207,214,192,212]
_R_UNDER_EYE = [339,425,280,427,434,416,432]


def analyze_skin(img_rgb: np.ndarray, pts: np.ndarray):
    """
    Full skin analysis using MediaPipe face landmarks.
    pts must be an (N,2) array of (x,y) pixel coords for 478 landmarks.
    Returns a dict with classification labels, scientific metrics, and base64 images.
    """
    try:
        h, w = img_rgb.shape[:2]

        def get_pts(indices):
            return np.array([(int(pts[i][0]), int(pts[i][1])) for i in indices], dtype=np.int32)

        face_pts   = get_pts(_FACE_OVAL)
        lips_pts   = get_pts(_LIPS)
        l_eye_pts  = get_pts(_LEFT_EYE)
        r_eye_pts  = get_pts(_RIGHT_EYE)
        l_brow_pts = get_pts(_LEFT_EYEBROW)
        r_brow_pts = get_pts(_RIGHT_EYEBROW)
        l_under    = get_pts(_L_UNDER_EYE)
        r_under    = get_pts(_R_UNDER_EYE)

        # Build face oval mask
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [face_pts], 255)

        # Exclude eyes / eyebrows / lips
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        for pts_feat in [lips_pts, l_eye_pts, r_eye_pts, l_brow_pts, r_brow_pts]:
            hull = cv2.convexHull(pts_feat)
            cv2.fillConvexPoly(exclude_mask, hull, 255)

        skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclude_mask))

        # Under-eye mask
        under_eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(under_eye_mask, [cv2.convexHull(l_under)], 255)
        cv2.fillPoly(under_eye_mask, [cv2.convexHull(r_under)], 255)

        # Convert to LAB
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lab_img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab_img)

        skin_pixels_l = l_ch[skin_mask == 255]
        skin_pixels_a = a_ch[skin_mask == 255]
        skin_pixels_b = b_ch[skin_mask == 255]

        mean_a = float(np.mean(skin_pixels_a))
        mean_b = float(np.mean(skin_pixels_b))
        mean_l = float(np.mean(skin_pixels_l))

        # ── Undertone ──
        b_offset = mean_b - 128
        a_offset = mean_a - 128
        if b_offset > 10 and a_offset < 10:
            undertone = "Warm"
        elif b_offset > 5 and a_offset < 5:
            undertone = "Neutral-Warm"
        elif a_offset > 10 and b_offset < 5:
            undertone = "Cool"
        elif a_offset > 5:
            undertone = "Neutral-Cool"
        else:
            undertone = "Neutral"

        # ── Redness / blemishing ──
        a_skin = cv2.bitwise_and(a_ch, a_ch, mask=skin_mask)
        _, red_mask = cv2.threshold(a_skin, mean_a + 15, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        blemish_count = sum(1 for i in range(1, num_labels) if 5 < stats[i, cv2.CC_STAT_AREA] < 500)
        if blemish_count < 3:   blemishing = "Clear"
        elif blemish_count < 10: blemishing = "Mild"
        elif blemish_count < 25: blemishing = "Moderate"
        else:                    blemishing = "Severe"

        # ── Roughness / Texture (RIN) ──
        gray_img   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian  = np.abs(cv2.Laplacian(gray_img, cv2.CV_64F))
        laplacian  = np.uint8(np.clip(laplacian, 0, 255))
        variance   = float(np.var(laplacian[skin_mask == 255]))
        rin_roughness = float(np.clip(0.05 + (variance / 3000.0), 0.05, 0.30))
        if rin_roughness < 0.10:   texture = "Smooth"
        elif rin_roughness < 0.14: texture = "Slightly Textured"
        else:                      texture = "Textured / Rough"

        # ── Oiliness (luminance skewness) ──
        luminance_skew = float(scipy_skew(skin_pixels_l))
        if luminance_skew > 0.30:   oiliness = "Oily / Shiny"
        elif luminance_skew > 0.0:  oiliness = "Normal / Combination"
        else:                        oiliness = "Matte / Dry"

        # ── Evenness (color std dev → homogeneity RIN) ──
        std_l = float(np.std(skin_pixels_l))
        std_a = float(np.std(skin_pixels_a))
        std_b = float(np.std(skin_pixels_b))
        total_std = (std_l + std_a + std_b) / 3.0
        homogeneity_rin = float(np.clip(0.10 + (total_std / 50.0), 0.10, 0.45))
        if homogeneity_rin < 0.20:   evenness = "Even"
        elif homogeneity_rin < 0.28: evenness = "Slightly Uneven"
        else:                         evenness = "Uneven"

        # ── Dark circles ──
        mean_l_under = float(np.mean(l_ch[under_eye_mask == 255])) if np.any(under_eye_mask == 255) else mean_l
        dark_circles = "Detected" if mean_l_under < mean_l * 0.9 else "Not Prominent"

        # ── Skin-only image (face oval, white background) ──
        skin_img_white = np.ones_like(img_rgb) * 255
        face_mask_bool = face_mask == 255
        skin_img_white[face_mask_bool] = img_rgb[face_mask_bool]
        # Also blank out the background outside face
        skin_img_b64 = rgb_to_b64(skin_img_white)

        return {
            # Classifications (for cards)
            "undertone":     undertone,
            "blemishing":    blemishing,
            "evenness":      evenness,
            "texture":       texture,
            "oiliness":      oiliness,
            "dark_circles":  dark_circles,
            # Scientific metrics
            "roughness_rin":     round(rin_roughness, 2),
            "homogeneity_rin":   round(homogeneity_rin, 2),
            "oiliness_skew":     round(luminance_skew, 2),
            "blemish_count":     blemish_count,
            "mean_redness_a":    round(mean_a, 1),
            "mean_luminance_l":  round(mean_l, 1),
            "under_eye_l":       round(mean_l_under, 1),
            # Image
            "skin_image": skin_img_b64,
        }
    except Exception as e:
        print(f"[WARN] Skin analysis failed: {e}")
        import traceback; traceback.print_exc()
        return {
            "undertone": "N/A", "blemishing": "N/A", "evenness": "N/A",
            "texture": "N/A", "oiliness": "N/A", "dark_circles": "N/A",
            "roughness_rin": None, "homogeneity_rin": None, "oiliness_skew": None,
            "blemish_count": None, "mean_redness_a": None, "mean_luminance_l": None,
            "under_eye_l": None, "skin_image": None,
        }


# Smile classifiers
def classify_smile_width(smile_w_mm):
    if smile_w_mm is None: return "N/A"
    if smile_w_mm < 40: return "Narrow"
    if smile_w_mm < 55: return "Average"
    return "Wide"

def classify_smile_shape(curvature):
    if curvature is None: return "N/A"
    if curvature < 0.02: return "Flat / Straight"
    if curvature < 0.08: return "Slightly Upturned"
    return "Strongly Upturned"

def classify_teeth_exposure():
    return "N/A"  # cannot be measured from static geometry

def classify_teeth_color():
    return "N/A"  # cannot be measured from geometry


# Neck classifiers
def classify_neck_width(neck_w_mm):
    if neck_w_mm is None: return "N/A"
    if neck_w_mm < 75: return "Slender"
    if neck_w_mm < 100: return "Average"
    return "Thick"

def classify_neck_definition():
    return "N/A"  # cannot be measured from landmarks

def classify_neck_length():
    return "N/A"  # cannot be measured from static segmentation alone

def classify_neck_aging():
    return "N/A"  # cannot be determined from geometry



# ─────────────────────────────────────────────
# Core metrics computation
# ─────────────────────────────────────────────

def compute_all_metrics(pts, labels, img_rgb):
    """
    Mirrors the full notebook metric extraction.
    Returns a dict of category -> {metric_name: value_or_None}.
    """
    h, w = img_rgb.shape[:2]

    # ── Reference landmarks ──
    FOREHEAD_TOP = 10
    GLABELLA     = 9
    NOSE_TIP     = 4
    NOSE_BRIDGE  = 168
    SUBNASALE    = 2
    MENTON       = 152
    R_ZYGION     = 234
    L_ZYGION     = 454
    R_TEMPLE     = 127
    L_TEMPLE     = 356
    R_GONION     = 172
    L_GONION     = 397

    # Eyebrow landmarks
    R_BROW_INNER, R_BROW_PEAK, R_BROW_OUTER = 55, 105, 70
    L_BROW_INNER, L_BROW_PEAK, L_BROW_OUTER = 285, 334, 300
    R_EYE_TOP, R_EYE_BOTTOM = 159, 145
    L_EYE_TOP, L_EYE_BOTTOM = 386, 374

    # Nose landmarks
    R_ALA, L_ALA = 129, 358
    R_BRIDGE_SIDE, L_BRIDGE_SIDE = 236, 456
    R_INNER_CANTHUS, L_INNER_CANTHUS = 133, 362

    # Lip landmarks
    MOUTH_R, MOUTH_L = 61, 291
    CUPID_DIP = 0
    CUPID_PEAK_R, CUPID_PEAK_L = 37, 267
    LOWER_LIP_BOTTOM = 17

    # Chin landmarks
    CHIN_R, CHIN_L = 214, 434

    try:
        face_width_px  = dist(pts, R_ZYGION, L_ZYGION)
        face_height_px = dist(pts, FOREHEAD_TOP, MENTON)
        ipd_px         = dist(pts, 33, 263)
        mm_per_px      = 63.5 / (ipd_px + 1e-9)
    except Exception:
        return {}

    def mm(px_val):
        return round(px_val * mm_per_px, 3) if px_val is not None else None

    def safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    # ── EYEBROWS ──
    r_peak_h_px = safe(lambda: abs(pts[R_BROW_PEAK][1] - pts[R_EYE_TOP][1]))
    l_peak_h_px = safe(lambda: abs(pts[L_BROW_PEAK][1] - pts[L_EYE_TOP][1]))
    r_elev_ratio = safe(lambda: abs(pts[R_BROW_PEAK][1] - pts[R_EYE_TOP][1]) / (ipd_px + 1e-9))
    l_elev_ratio = safe(lambda: abs(pts[L_BROW_PEAK][1] - pts[L_EYE_TOP][1]) / (ipd_px + 1e-9))
    r_apex_angle = safe(angle_deg, pts, R_BROW_PEAK, R_BROW_INNER, R_BROW_OUTER)
    l_apex_angle = safe(angle_deg, pts, L_BROW_PEAK, L_BROW_INNER, L_BROW_OUTER)
    avg_height_mm = mm((r_peak_h_px + l_peak_h_px) / 2) if r_peak_h_px and l_peak_h_px else None
    avg_elev = round((r_elev_ratio + l_elev_ratio) / 2, 4) if r_elev_ratio and l_elev_ratio else None
    avg_apex = round((r_apex_angle + l_apex_angle) / 2, 2) if r_apex_angle and l_apex_angle else None

    eyebrow_metrics = {
        "right_brow_peak_height_mm": mm(r_peak_h_px),
        "left_brow_peak_height_mm":  mm(l_peak_h_px),
        "right_brow_elevation_ratio": r_elev_ratio,
        "left_brow_elevation_ratio":  l_elev_ratio,
        "right_brow_apex_angle_deg": r_apex_angle,
        "left_brow_apex_angle_deg":  l_apex_angle,
        # derived
        "avg_height_mm":    avg_height_mm,
        "avg_elevation":    avg_elev,
        "avg_apex_angle":   avg_apex,
        "position_class":   classify_brow_position(avg_height_mm),
        "tilt_class":       classify_brow_tilt(avg_apex),
        "shape_class":      classify_brow_shape(0.08),  # placeholder curvature
        "virility_class":   classify_brow_virility(avg_elev),
    }

    # ── EYES ──
    r_ear = safe(eye_aspect_ratio, pts, 33, 160, 158, 133, 153, 144)
    l_ear = safe(eye_aspect_ratio, pts, 362, 385, 387, 263, 373, 380)
    avg_ear = round((r_ear + l_ear) / 2, 4) if r_ear and l_ear else None
    spacing_ratio = round(ipd_px / (face_width_px + 1e-9), 4)
    r_lid_curv = safe(curvature_ratio, pts, 33, R_EYE_BOTTOM, 133)
    l_lid_curv = safe(curvature_ratio, pts, 362, L_EYE_BOTTOM, 263)
    avg_curv = round((r_lid_curv + l_lid_curv) / 2, 4) if r_lid_curv and l_lid_curv else None

    eye_metrics = {
        "right_eye_aspect_ratio": r_ear,
        "left_eye_aspect_ratio":  l_ear,
        "avg_ear":                avg_ear,
        "eye_spacing_ratio_ipd_over_face_width": spacing_ratio,
        "right_lower_eyelid_curvature": r_lid_curv,
        "left_lower_eyelid_curvature":  l_lid_curv,
        "avg_lower_eyelid_curvature":   avg_curv,
        # classes
        "tilt_class":     classify_eye_tilt(pts),
        "exposure_class": classify_eyelid_exposure(avg_ear),
        "sclera_class":   classify_sclera(),
        "health_class":   classify_under_eye(),
    }

    # ── NOSE ──
    nose_width_px   = safe(dist, pts, R_ALA, L_ALA)
    nose_height_px  = safe(dist, pts, NOSE_BRIDGE, SUBNASALE)
    intercanthal_px = safe(dist, pts, R_INNER_CANTHUS, L_INNER_CANTHUS)
    pyramidal_px    = safe(dist, pts, R_BRIDGE_SIDE, L_BRIDGE_SIDE)
    nose_ar  = round(nose_width_px / (nose_height_px + 1e-9), 4) if nose_width_px and nose_height_px else None
    naso_canthal = round(nose_width_px / (intercanthal_px + 1e-9), 4) if nose_width_px and intercanthal_px else None

    nose_metrics = {
        "nasal_width_mm":         mm(nose_width_px),
        "nasal_height_mm":        mm(nose_height_px),
        "nasal_aspect_ratio_w_over_h": nose_ar,
        "naso_canthal_ratio":     naso_canthal,
        "pyramidal_width_mm":     mm(pyramidal_px),
        # classes
        "shape":  classify_nose_shape(nose_ar),
        "height": classify_nose_height(mm(nose_height_px)),
        "tip":    classify_nose_tip(naso_canthal),
        "width":  classify_nose_width(mm(nose_width_px)),
    }

    # ── LIPS ──
    mouth_width_px  = safe(dist, pts, MOUTH_R, MOUTH_L)
    philtrum_px     = safe(dist, pts, SUBNASALE, CUPID_DIP)
    cupid_angle     = safe(angle_deg, pts, CUPID_DIP, CUPID_PEAK_R, CUPID_PEAK_L)

    lips_metrics = {
        "mouth_width_mm":      mm(mouth_width_px),
        "philtrum_length_mm":  mm(philtrum_px),
        "cupids_bow_angle_deg": cupid_angle,
        # classes
        "fullness":    classify_lip_fullness(mm(mouth_width_px), mm(philtrum_px)),
        "width":       "Wide" if mm(mouth_width_px) and mm(mouth_width_px) > 55 else ("Narrow" if mm(mouth_width_px) and mm(mouth_width_px) < 40 else "Average"),
        "proportions": classify_lip_proportions(cupid_angle),
        "health":      "N/A",
    }

    # ── CHEEKS ──
    cheekbone_pos_r = round((pts[R_ZYGION][1] - pts[FOREHEAD_TOP][1]) / (face_height_px + 1e-9), 4) if face_height_px else None
    cheekbone_pos_l = round((pts[L_ZYGION][1] - pts[FOREHEAD_TOP][1]) / (face_height_px + 1e-9), 4) if face_height_px else None
    malar_ratio     = round(face_width_px / (face_height_px + 1e-9), 4) if face_height_px else None

    cheeks_metrics = {
        "facial_width_mm":                mm(face_width_px),
        "malar_width_ratio":              malar_ratio,
        "right_cheekbone_vertical_position_ratio": cheekbone_pos_r,
        "left_cheekbone_vertical_position_ratio":  cheekbone_pos_l,
        # classes
        "width_class":   classify_cheek_width(malar_ratio),
        "position":      classify_cheek_position(cheekbone_pos_r),
        "fullness":      classify_cheek_fullness(malar_ratio),
        "height":        "N/A",
        "width_val":     mm(face_width_px),
    }

    # ── JAW ──
    jaw_width_px = safe(dist, pts, R_GONION, L_GONION)
    r_jaw_angle  = safe(angle_from_horizontal, pts, R_GONION, MENTON)
    l_jaw_angle  = safe(angle_from_horizontal, pts, L_GONION, MENTON)
    jaw_rise_px  = safe(lambda: abs(pts[R_GONION][1] - pts[MENTON][1]))

    jaw_metrics = {
        "jaw_width_mm":                mm(jaw_width_px),
        "frontal_jaw_rise_mm":         mm(jaw_rise_px),
        "right_jaw_inclination_angle_deg": r_jaw_angle,
        "left_jaw_inclination_angle_deg":  l_jaw_angle,
        "face_width_mm":               mm(face_width_px),
    }

    # ── CHIN ──
    chin_width_px  = safe(dist, pts, CHIN_R, CHIN_L)
    chin_height_px = safe(dist, pts, LOWER_LIP_BOTTOM, MENTON)
    chin_dev_px    = safe(point_line_deviation, pts, MENTON, GLABELLA, NOSE_TIP)

    chin_metrics = {
        "chin_width_mm":              mm(chin_width_px),
        "chin_vertical_height_mm":    mm(chin_height_px),
        "chin_midline_deviation_mm":  mm(chin_dev_px),
    }

    # ── HAIR / FOREHEAD ──
    forehead_width_px  = safe(dist, pts, R_TEMPLE, L_TEMPLE)
    forehead_height_px = safe(dist, pts, FOREHEAD_TOP, GLABELLA)
    r_temple_angle     = safe(angle_from_vertical, pts, R_TEMPLE, R_ZYGION)
    l_temple_angle     = safe(angle_from_vertical, pts, L_TEMPLE, L_ZYGION)

    hair_metrics = {
        "forehead_width_mm":               mm(forehead_width_px),
        "forehead_height_mm":              mm(forehead_height_px),
        "right_temple_inclination_angle_deg": r_temple_angle,
        "left_temple_inclination_angle_deg":  l_temple_angle,
    }

    # ── SMILE ──
    upper_curv = safe(curvature_ratio, pts, MOUTH_R, CUPID_DIP, MOUTH_L)
    lower_curv = safe(curvature_ratio, pts, MOUTH_R, LOWER_LIP_BOTTOM, MOUTH_L)
    smile_width_mm = mm(mouth_width_px)

    smile_metrics = {
        "upper_smile_arc_curvature": upper_curv,
        "lower_smile_arc_curvature": lower_curv,
        "smile_width_mm":            smile_width_mm,
    }

    # ── NECK (from segmentation mask) ──
    neck_mask = (labels == 17)
    neck_pixels = np.where(neck_mask)
    if len(neck_pixels[0]) > 0:
        ys_n, xs_n = neck_pixels
        neck_width_px_seg = int(xs_n.max()) - int(xs_n.min())
        neck_metrics = {
            "neck_width_mm":               mm(neck_width_px_seg),
            "neck_width_to_jaw_width_ratio": round(neck_width_px_seg / (jaw_width_px + 1e-9), 4) if jaw_width_px else None,
        }
    else:
        neck_metrics = {
            "neck_width_mm":               None,
            "neck_width_to_jaw_width_ratio": None,
        }

    return {
        "eyebrow": eyebrow_metrics,
        "eye":     eye_metrics,
        "nose":    nose_metrics,
        "lips":    lips_metrics,
        "cheeks":  cheeks_metrics,
        "jaw":     jaw_metrics,
        "chin":    chin_metrics,
        "hair":    hair_metrics,
        "smile":   smile_metrics,
        "neck":    neck_metrics,
        "meta": {
            "face_width_mm":  mm(face_width_px),
            "face_height_mm": mm(face_height_px),
            "ipd_px":         round(ipd_px, 2),
            "mm_per_px":      round(mm_per_px, 6),
            "img_width":      w,
            "img_height":     h,
        },
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/eyes")
def eyes():
    return render_template("eyes.html")


@app.route("/eyebrows")
def eyebrows():
    return render_template("eyebrows.html")


@app.route("/nose")
def nose():
    return render_template("nose.html")


@app.route("/lips")
def lips():
    return render_template("lips.html")


@app.route("/cheeks")
def cheeks():
    return render_template("cheeks.html")


@app.route("/jaw")
def jaw():
    return render_template("jaw.html")


@app.route("/chin")
def chin():
    return render_template("chin.html")


@app.route("/hair")
def hair():
    return render_template("hair.html")


@app.route("/smile")
def smile():
    return render_template("smile.html")


@app.route("/neck")
def neck():
    return render_template("neck.html")


@app.route("/ear")
def ear():
    return render_template("ear.html")


@app.route("/skin")
def skin():
    return render_template("skin.html")


def extract_ear_roboflow(pil_img: Image.Image, mm_per_px: float = None):
    """
    Uses Roboflow API to segment the ear from a side-face image.
    Returns:
      - cropped_b64: cropped ear on white background
      - overlay_b64: full image with ear outline + caliper measurement lines
      - metrics: dict with ear_height_mm, ear_width_mm, ear_height_px, ear_width_px
    """
    import os
    try:
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            pil_img.save(tmp_path)

        result = _roboflow_client.run_workflow(
            workspace_name="fabiki4429-acoxs-com",
            workflow_id="general-segmentation-api-2",
            images={"image": tmp_path},
            parameters={"classes": "ear"},
            use_cache=True
        )
        os.unlink(tmp_path)

        preds = result[0].get('predictions', {})
        if isinstance(preds, dict):
            predictions_list = preds.get('predictions', [])
        else:
            predictions_list = []

        if not predictions_list:
            return None, None, {}

        prediction = predictions_list[0]
        points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32)

        img_rgb = np.array(pil_img.convert('RGB'))

        # --- Cropped ear on white background ---
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        ear_masked = img_rgb.copy()
        ear_masked[mask == 0] = 255  # white background

        xs, ys = points[:, 0], points[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        pad = 10
        x1 = max(0, int(min_x) - pad)
        y1 = max(0, int(min_y) - pad)
        x2 = min(img_rgb.shape[1], int(max_x) + pad)
        y2 = min(img_rgb.shape[0], int(max_y) + pad)
        ear_crop = ear_masked[y1:y2, x1:x2]
        cropped_b64 = rgb_to_b64(ear_crop)

        # --- Full image with caliper lines ---
        def _draw_capped_line(draw, p1, p2, color, width, cap_len):
            draw.line([p1, p2], fill=color, width=width)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length == 0: return
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            half = cap_len / 2
            for (cx, cy) in (p1, p2):
                a = (cx - px * half, cy - py * half)
                b = (cx + px * half, cy + py * half)
                draw.line([a, b], fill=color, width=width)

        overlay_img = pil_img.convert('RGB').copy()
        draw = ImageDraw.Draw(overlay_img)
        pts_list = [tuple(p) for p in points.tolist()]
        draw.line(pts_list + [pts_list[0]], fill=(0, 200, 80), width=3, joint='curve')

        pts_plain = points.tolist()
        vx = int(min_x) - 18
        _draw_capped_line(draw, (vx, int(min_y)), (vx, int(max_y)), (237, 234, 222), 2, 14)

        hy = int(min_y + (max_y - min_y) * 0.20)
        band = [p for p in pts_plain if abs(p[1] - hy) < (max_y - min_y) * 0.05]
        if band:
            hx_left = min(p[0] for p in band)
            hx_right = max(p[0] for p in band)
        else:
            hx_left, hx_right = int(min_x), int(max_x)
        _draw_capped_line(draw, (int(hx_left), hy), (int(hx_right), hy), (237, 234, 222), 2, 14)

        top_pt = min(pts_plain, key=lambda p: p[1])
        bottom_pt = max(pts_plain, key=lambda p: p[1])
        _draw_capped_line(draw, (int(top_pt[0]), int(top_pt[1])), (int(bottom_pt[0]), int(bottom_pt[1])), (237, 234, 222), 2, 14)

        overlay_arr = np.array(overlay_img)
        overlay_b64 = rgb_to_b64(overlay_arr)

        ear_h_px = int(max_y - min_y)
        ear_w_px = int(hx_right - hx_left)
        metrics = {
            'ear_height_px': ear_h_px,
            'ear_width_px': ear_w_px,
            'ear_height_mm': round(ear_h_px * mm_per_px, 2) if mm_per_px else None,
            'ear_width_mm': round(ear_w_px * mm_per_px, 2) if mm_per_px else None,
        }
        return cropped_b64, overlay_b64, metrics

    except Exception as e:
        print(f'[WARN] Roboflow ear extraction failed: {e}')
        return None, None, {}


@app.route("/analyze_all", methods=["POST"])
def analyze_all():
    """
    Receives a front-face image, runs segmentation + landmark detection,
    computes all metrics, and returns a single JSON with:
     - eyes, eyebrows, nose, lips, cheeks, jaw, chin, hair, smile, neck keys
     - each key contains metrics + part images (white_bg + cropped) as base64
    """
    if "front" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["front"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
        img_rgb = pil_to_rgb(pil_img)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        # ── Segmentation ──
        labels = run_segmentation(pil_img)

        # ── Landmarks ──
        pts, found = run_landmarks(img_rgb)
        if not found:
            return jsonify({"error": "No face detected. Please upload a clear frontal face photo."}), 400

        # ── Metrics ──
        metrics = compute_all_metrics(pts, labels, img_rgb)

        # ── Part images ──
        # Each label index per LABEL_MAP
        def get_images(idx):
            wb, cr = part_images_b64(img_rgb, labels, idx)
            return {"white_bg": wb, "cropped": cr or wb}  # fallback to white_bg if crop fails

        part_imgs = {name: get_images(idx) for idx, name in LABEL_MAP.items()}
        part_imgs["combined_lips"] = get_images([11, 12])  # u_lip and l_lip combined
        
        # Chin via Mediapipe
        wb_chin, cr_chin = extract_chin_mediapipe(img_rgb, pts)
        part_imgs["chin_mediapipe"] = {
            "white_bg": rgb_to_b64(wb_chin) if wb_chin is not None else None,
            "cropped": rgb_to_b64(cr_chin) if cr_chin is not None else None,
        }

        # Cheeks via Mediapipe
        wb_cheeks, cr_cheeks = extract_cheeks_mediapipe(img_rgb, pts)
        part_imgs["cheeks_mediapipe"] = {
            "white_bg": rgb_to_b64(wb_cheeks) if wb_cheeks is not None else None,
            "cropped": rgb_to_b64(cr_cheeks) if cr_cheeks is not None else None,
        }

        # ── Hair Color Analysis ──
        hair_mask = (labels == 13)
        hair_color_analysis = analyze_hair_color(img_rgb, hair_mask)

        # ── Skin Analysis ──
        skin_data = analyze_skin(img_rgb, pts)


        em = metrics.get("eyebrow", {})
        eyebrows_data = {
            # numeric metrics
            "right_brow_peak_height_mm":   _fmt(em.get("right_brow_peak_height_mm")),
            "left_brow_peak_height_mm":    _fmt(em.get("left_brow_peak_height_mm")),
            "right_brow_elevation_ratio":  _fmt(em.get("right_brow_elevation_ratio"), 4),
            "left_brow_elevation_ratio":   _fmt(em.get("left_brow_elevation_ratio"), 4),
            "right_brow_apex_angle_deg":   _fmt(em.get("right_brow_apex_angle_deg")),
            "left_brow_apex_angle_deg":    _fmt(em.get("left_brow_apex_angle_deg")),
            "avg_height_mm":               _fmt(em.get("avg_height_mm")),
            "avg_elevation":               _fmt(em.get("avg_elevation"), 4),
            "avg_apex_angle":              _fmt(em.get("avg_apex_angle")),
            # classes
            "position_class": em.get("position_class", "N/A"),
            "tilt_class":     em.get("tilt_class", "N/A"),
            "shape_class":    em.get("shape_class", "N/A"),
            "virility_class": em.get("virility_class", "N/A"),
            # images (right brow = label 7, left brow = label 6)
            "r_brow_image_white": part_imgs["r_brow"]["white_bg"],
            "r_brow_image":       part_imgs["r_brow"]["cropped"],
            "l_brow_image_white": part_imgs["l_brow"]["white_bg"],
            "l_brow_image":       part_imgs["l_brow"]["cropped"],
        }

        eym = metrics.get("eye", {})
        eyes_data = {
            "right_eye_aspect_ratio":          _fmt(eym.get("right_eye_aspect_ratio"), 4),
            "left_eye_aspect_ratio":           _fmt(eym.get("left_eye_aspect_ratio"), 4),
            "avg_ear":                         _fmt(eym.get("avg_ear"), 4),
            "eye_spacing_ratio":               _fmt(eym.get("eye_spacing_ratio_ipd_over_face_width"), 4),
            "right_lower_eyelid_curvature":    _fmt(eym.get("right_lower_eyelid_curvature"), 4),
            "left_lower_eyelid_curvature":     _fmt(eym.get("left_lower_eyelid_curvature"), 4),
            "avg_lower_eyelid_curvature":      _fmt(eym.get("avg_lower_eyelid_curvature"), 4),
            # classes (match existing HTML IDs)
            "tilt_class":     eym.get("tilt_class", "N/A"),
            "exposure_class": eym.get("exposure_class", "N/A"),
            "sclera_class":   eym.get("sclera_class", "N/A"),
            "health_class":   eym.get("health_class", "N/A"),
            # carousel values
            "curvature":     float(eym.get("avg_lower_eyelid_curvature") or 0),
            "ear":           float(eym.get("avg_ear") or 0),
            "spacing_ratio": float(eym.get("eye_spacing_ratio_ipd_over_face_width") or 0),
            # images (r_eye=5, l_eye=4)
            "r_eye_image_white": part_imgs["r_eye"]["white_bg"],
            "r_eye_image":       part_imgs["r_eye"]["cropped"],
            "l_eye_image_white": part_imgs["l_eye"]["white_bg"],
            "l_eye_image":       part_imgs["l_eye"]["cropped"],
        }

        nm = metrics.get("nose", {})
        nose_data = {
            "nasal_width_mm":              _fmt(nm.get("nasal_width_mm")),
            "nasal_height_mm":             _fmt(nm.get("nasal_height_mm")),
            "nasal_aspect_ratio":          _fmt(nm.get("nasal_aspect_ratio_w_over_h"), 4),
            "naso_canthal_ratio":          _fmt(nm.get("naso_canthal_ratio"), 4),
            "pyramidal_width_mm":          _fmt(nm.get("pyramidal_width_mm")),
            "shape":  nm.get("shape", "N/A"),
            "height": nm.get("height", "N/A"),
            "tip":    nm.get("tip", "N/A"),
            "width":  nm.get("width", "N/A"),
            "ratio":  _fmt(nm.get("nasal_aspect_ratio_w_over_h"), 4),
            # images (nose=2)
            "nose_image_white": part_imgs["nose"]["white_bg"],
            "nose_image":       part_imgs["nose"]["cropped"],
        }

        lm_data = metrics.get("lips", {})
        lips_data = {
            "mouth_width_mm":       _fmt(lm_data.get("mouth_width_mm")),
            "philtrum_length_mm":   _fmt(lm_data.get("philtrum_length_mm")),
            "cupids_bow_angle_deg": _fmt(lm_data.get("cupids_bow_angle_deg")),
            "fullness":    lm_data.get("fullness", "N/A"),
            "width":       lm_data.get("width", "N/A"),
            "proportions": lm_data.get("proportions", "N/A"),
            "health":      lm_data.get("health", "N/A"),
            "mouth_width": float(lm_data.get("mouth_width_mm") or 0),
            # images: upper_lip=11, lower_lip=12, mouth=10, combined_lips=[11, 12]
            "lip_image_white":       part_imgs["combined_lips"]["white_bg"],
            "lip_image":             part_imgs["combined_lips"]["cropped"],
            "lower_lip_image_white": part_imgs["l_lip"]["white_bg"],
            "lower_lip_image":       part_imgs["l_lip"]["cropped"],
            "mouth_image_white":     part_imgs["mouth"]["white_bg"],
            "mouth_image":           part_imgs["mouth"]["cropped"],
        }

        cm = metrics.get("cheeks", {})
        cheeks_data = {
            "facial_width_mm":            _fmt(cm.get("facial_width_mm")),
            "malar_width_ratio":          _fmt(cm.get("malar_width_ratio"), 4),
            "right_cheekbone_pos_ratio":  _fmt(cm.get("right_cheekbone_vertical_position_ratio"), 4),
            "left_cheekbone_pos_ratio":   _fmt(cm.get("left_cheekbone_vertical_position_ratio"), 4),
            "width_class":  cm.get("width_class", "N/A"),
            "position":     cm.get("position", "N/A"),
            "fullness":     cm.get("fullness", "N/A"),
            "height":       cm.get("height", "N/A"),
            "width_val":    float(cm.get("width_val") or 0),
            # images (extracted via mediapipe)
            "cheeks_image_white": part_imgs["cheeks_mediapipe"]["white_bg"],
            "cheeks_image":       part_imgs["cheeks_mediapipe"]["cropped"],
        }

        jm = metrics.get("jaw", {})
        jaw_data = {
            "jaw_width_mm":                    _fmt(jm.get("jaw_width_mm")),
            "frontal_jaw_rise_mm":             _fmt(jm.get("frontal_jaw_rise_mm")),
            "right_jaw_inclination_angle_deg": _fmt(jm.get("right_jaw_inclination_angle_deg")),
            "left_jaw_inclination_angle_deg":  _fmt(jm.get("left_jaw_inclination_angle_deg")),
            "face_width_mm":                   _fmt(jm.get("face_width_mm")),
            # classifications
            "definition": classify_jaw_definition(jm.get("frontal_jaw_rise_mm")),
            "width_class": classify_jaw_width(jm.get("jaw_width_mm")),
            "shape":       classify_jaw_shape(jm.get("right_jaw_inclination_angle_deg"), jm.get("left_jaw_inclination_angle_deg")),
            "angle":       classify_jaw_angle(jm.get("right_jaw_inclination_angle_deg"), jm.get("left_jaw_inclination_angle_deg")),
            # images (skin label=1 used as full face proxy for jaw)
            "jaw_image_white": part_imgs["skin"]["white_bg"],
            "jaw_image":       part_imgs["skin"]["cropped"],
        }

        chm = metrics.get("chin", {})
        chin_data = {
            "chin_width_mm":             _fmt(chm.get("chin_width_mm")),
            "chin_vertical_height_mm":   _fmt(chm.get("chin_vertical_height_mm")),
            "chin_midline_deviation_mm": _fmt(chm.get("chin_midline_deviation_mm")),
            # classifications
            "width_class":   classify_chin_width(chm.get("chin_width_mm")),
            "projection":    classify_chin_projection(chm.get("chin_midline_deviation_mm")),
            "shape":         classify_chin_shape(chm.get("chin_width_mm"), chm.get("chin_vertical_height_mm")),
            "depth":         classify_chin_depth(chm.get("chin_vertical_height_mm")),
            # images (extracted via mediapipe)
            "chin_image_white": part_imgs["chin_mediapipe"]["white_bg"],
            "chin_image":       part_imgs["chin_mediapipe"]["cropped"],
        }

        hm = metrics.get("hair", {})
        hair_data = {
            "forehead_width_mm":                  _fmt(hm.get("forehead_width_mm")),
            "forehead_height_mm":                 _fmt(hm.get("forehead_height_mm")),
            "right_temple_inclination_angle_deg": _fmt(hm.get("right_temple_inclination_angle_deg")),
            "left_temple_inclination_angle_deg":  _fmt(hm.get("left_temple_inclination_angle_deg")),
            # classifications
            "temple_width_class": classify_temple_width(hm.get("forehead_width_mm")),
            "hair_volume":        classify_hair_volume(),
            "hair_density":       classify_hair_density(),
            "hairline_shape":     classify_hairline_shape(),
            # images (hair=13)
            "hair_image_white": part_imgs["hair"]["white_bg"],
            "hair_image":       part_imgs["hair"]["cropped"],
            # color analysis
            **hair_color_analysis,
        }

        sm = metrics.get("smile", {})
        smile_data = {
            "upper_smile_arc_curvature": _fmt(sm.get("upper_smile_arc_curvature"), 4),
            "lower_smile_arc_curvature": _fmt(sm.get("lower_smile_arc_curvature"), 4),
            "smile_width_mm":            _fmt(sm.get("smile_width_mm")),
            # classifications
            "mouth_width_class": classify_smile_width(sm.get("smile_width_mm")),
            "smile_shape":       classify_smile_shape(sm.get("upper_smile_arc_curvature")),
            "teeth_exposure":    classify_teeth_exposure(),
            "teeth_color":       classify_teeth_color(),
            # images (using combined lips as requested)
            "smile_image_white": part_imgs["combined_lips"]["white_bg"],
            "smile_image":       part_imgs["combined_lips"]["cropped"],
        }

        neckm = metrics.get("neck", {})
        neck_data = {
            "neck_width_mm":               _fmt(neckm.get("neck_width_mm")),
            "neck_width_to_jaw_width_ratio": _fmt(neckm.get("neck_width_to_jaw_width_ratio"), 4),
            # classifications
            "width_class":    classify_neck_width(neckm.get("neck_width_mm")),
            "definition":     classify_neck_definition(),
            "length":         classify_neck_length(),
            "aging":          classify_neck_aging(),
            # images (neck=17)
            "neck_image_white": part_imgs["neck"]["white_bg"],
            "neck_image":       part_imgs["neck"]["cropped"],
        }

        meta = metrics.get("meta", {})

        return jsonify({
            "eyebrows": eyebrows_data,
            "eyes":     eyes_data,
            "nose":     nose_data,
            "lips":     lips_data,
            "cheeks":   cheeks_data,
            "jaw":      jaw_data,
            "chin":     chin_data,
            "hair":     hair_data,
            "smile":    smile_data,
            "neck":     neck_data,
            "meta":     meta,
            # all 19 part images for a full face parsing view
            "all_parts": {
                name: {"white_bg": part_imgs[name]["white_bg"], "cropped": part_imgs[name]["cropped"]}
                for name in part_imgs
            },
            "skin":     skin_data,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_ear", methods=["POST"])
def analyze_ear():
    """
    Separate endpoint for ear analysis using a side-face image via Roboflow.
    """
    if "side" not in request.files:
        return jsonify({"error": "No side image uploaded"}), 400

    file = request.files["side"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        # Try to get mm_per_px from query param if passed
        mm_per_px_str = request.form.get('mm_per_px', None)
        mm_per_px = float(mm_per_px_str) if mm_per_px_str else None

        cropped_b64, overlay_b64, metrics = extract_ear_roboflow(pil_img, mm_per_px)

        if cropped_b64 is None:
            return jsonify({"error": "No ear detected in the image. Please upload a clear side-profile photo."}), 400

        ear_h_px = metrics.get('ear_height_px', 0)
        ear_w_px = metrics.get('ear_width_px', 0)
        ear_h_mm = metrics.get('ear_height_mm')
        ear_w_mm = metrics.get('ear_width_mm')

        def classify_ear_size(h_mm):
            if h_mm is None: return 'N/A'
            if h_mm < 55: return 'Small'
            if h_mm < 68: return 'Average'
            return 'Large'

        def classify_ear_prominence(w_mm):
            if w_mm is None: return 'N/A'
            if w_mm < 18: return 'Close-Set'
            if w_mm < 30: return 'Average'
            return 'Prominent'

        def classify_ear_shape(h_px, w_px):
            if h_px == 0: return 'N/A'
            ratio = w_px / h_px if h_px else 0
            if ratio < 0.45: return 'Narrow / Oval'
            if ratio < 0.65: return 'Rounded'
            return 'Wide'

        def classify_ear_position():
            return 'Mid-Set'  # cannot compute from single image without front-face reference

        ear_data = {
            'ear_height_mm': _fmt(ear_h_mm) if ear_h_mm else 'N/A',
            'ear_width_mm':  _fmt(ear_w_mm) if ear_w_mm else 'N/A',
            'ear_height_px': ear_h_px,
            'ear_width_px':  ear_w_px,
            'ear_size':        classify_ear_size(ear_h_mm),
            'ear_prominence':  classify_ear_prominence(ear_w_mm),
            'ear_shape':       classify_ear_shape(ear_h_px, ear_w_px),
            'ear_position':    classify_ear_position(),
            'ear_cropped':     cropped_b64,
            'ear_overlay':     overlay_b64,
        }
        return jsonify({'ear': ear_data})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/analyze_jaw", methods=["POST"])
def analyze_jaw():
    """
    Separate endpoint for jaw analysis using a 45-degree side-face image via MediaPipe.
    """
    if "jaw" not in request.files:
        return jsonify({"error": "No jaw image uploaded"}), 400

    file = request.files["jaw"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        dots_b64, highlight_b64 = extract_jaw_mediapipe(pil_img)

        if dots_b64 is None:
            return jsonify({"error": "No face detected by MediaPipe. Please ensure it's a 45-degree or frontal image!"}), 400

        jaw_data = {
            'jaw_cropped': dots_b64,
            'jaw_overlay': highlight_b64,
        }
        return jsonify({'jaw': jaw_data})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    load_models()
    app.run(debug=False, host="0.0.0.0", port=5000)
