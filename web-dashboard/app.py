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

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
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
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    global _segformer_processor, _segformer_model, _face_mesh
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


# ─────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────

def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def rgb_to_b64(arr: np.ndarray) -> str:
    """Convert RGB numpy array to base64 PNG data URI."""
    img_pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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

        # ── Build per-feature response dicts ──
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
            # images (skin=1 for cheeks region)
            "cheeks_image_white": part_imgs["skin"]["white_bg"],
            "cheeks_image":       part_imgs["skin"]["cropped"],
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
            # images (mouth area proxy)
            "chin_image_white": part_imgs["skin"]["white_bg"],
            "chin_image":       part_imgs["skin"]["cropped"],
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
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    load_models()
    app.run(debug=False, host="0.0.0.0", port=5000)
