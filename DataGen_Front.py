import cv2
import mediapipe as mp
import numpy as np
import os
import random

# ==========================================
#             CONFIGURATION
# ==========================================

DATASET_DIR = "dataset_dynamic_3d"

# Video / capture source
# Use an integer for webcam index (0, 1, ...) or a string path for a video file.
# Examples:
#   CAPTURE_SOURCE = 0
#   CAPTURE_SOURCE = "C:/path/to/video.mp4"
CAPTURE_SOURCE = 0

# --- DYNAMIC CURVE CONFIGURATION ---

# 1. Base Values (At mid-range distance, mild pose)
BASE_CLOSED_THRESH = 0.10   # around scale ~0.16, mild angles
BASE_OPEN_THRESH   = 0.21

# 2. Maximum Limits (Hard Caps)
MAX_CLOSED_THRESH = 0.150
MIN_OPEN_THRESH   = 0.180

# 3. ANGLE CURVE CONFIG
# We use a non-linear curve on combined |yaw|+|pitch| so that
# small angles barely move the threshold, but extreme angles raise it more.
ANGLE_MAX_MAG = 1.2          # combined |yaw|+|pitch| where we hit full angle effect
ANGLE_CLOSED_BOOST_MAX = 0.015  # extra added to closed thresh at ANGLE_MAX_MAG
ANGLE_OPEN_DROP_MAX   = 0.020  # amount we (slightly) lower open thresh at ANGLE_MAX_MAG

# 4. SCALE CURVE CONFIG (Face distance)
# We describe a smooth curve in (scale_ratio -> closed_threshold) space
# that approximately hits your requested anchor points:
#   scale 0.09 -> ~0.06
#   scale 0.16 -> ~0.10
#   scale 0.24 -> ~0.135
SCALE_REF_NEAR  = 0.04
SCALE_REF_MID   = 0.11
SCALE_REF_FAR   = 0.22
SCALE_CLOSED_NEAR = 0.030
SCALE_CLOSED_MID  = 0.110
SCALE_CLOSED_FAR  = 0.125

# Open threshold will track the closed curve with a fixed band.
OPEN_BAND_OFFSET = 0.09   # open_thresh ~= closed_thresh + this (then clamped)

# 5. SCALE REFERENCE (for display only)
OPTIMAL_FACE_RATIO = SCALE_REF_MID

# --- Dataset Balancing ---
OPEN_EYE_SAVE_PROB = 0 

# --- Image Quality ---
BLUR_THRESHOLD = 10       
PATCH_SIZE = (64, 64)     
SQUARE_PAD_RATIO = 0.8    

# --- Output Settings ---
SAVE_SEPARATE_EYES = True 
SAVE_COMBINED = True      

# ==========================================
#           END CONFIGURATION
# ==========================================

for state in ["open", "closed"]:
    os.makedirs(os.path.join(DATASET_DIR, state), exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10

def get_3d_point(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def compute_3D_EAR(landmarks, eye_indices, w, h):
    p1 = get_3d_point(landmarks[eye_indices[0]], w, h)
    p2 = get_3d_point(landmarks[eye_indices[1]], w, h)
    p3 = get_3d_point(landmarks[eye_indices[2]], w, h)
    p4 = get_3d_point(landmarks[eye_indices[3]], w, h)
    p5 = get_3d_point(landmarks[eye_indices[4]], w, h)
    p6 = get_3d_point(landmarks[eye_indices[5]], w, h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0: return 0.0, []
    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    pts_2d = [(int(p[0]), int(p[1])) for p in [p1, p2, p3, p4, p5, p6]]
    return EAR, pts_2d

def get_head_pose_ratios(landmarks):
    nose = landmarks[NOSE_TIP]
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    
    # --- YAW ---
    eye_mid_x = (left_outer.x + right_outer.x) / 2
    face_width = abs(right_outer.x - left_outer.x)
    yaw_ratio = (nose.x - eye_mid_x) / (face_width + 1e-6)
    
    # --- PITCH ---
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    face_height = abs(chin.y - forehead.y)
    face_mid_y = (forehead.y + chin.y) / 2
    pitch_ratio = (nose.y - face_mid_y) / (face_height + 1e-6)
    
    return yaw_ratio, pitch_ratio

def get_face_scale_ratio(landmarks):
    """
    Returns the ratio of 'Face Width' to 'Image Width'.
    Used to estimate how close the user is to the camera.
    """
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    # Simple Euclidean distance in normalized coordinates (0.0 to 1.0)
    # Since landmarks.x is already normalized by image width, this IS the ratio.
    return abs(right_outer.x - left_outer.x)

def angle_factor(angle_mag):
    """Smooth non-linear factor in [0,1] from combined |yaw|+|pitch|.

    Very small angles give almost 0, large angles asymptotically approach 1.
    """
    # Exponential saturation curve: f = 1 - exp(-k * x)
    k = 1.5
    x = max(angle_mag, 0.0)
    raw = 1.0 - np.exp(-k * x)
    max_raw = 1.0 - np.exp(-k * ANGLE_MAX_MAG)
    if max_raw <= 0:
        return 0.0
    return np.clip(raw / max_raw, 0.0, 1.0)

def closed_thresh_from_scale(scale):
    """Piecewise-smooth curve for closed-eye threshold vs face scale.

    Uses simple interpolation between three anchor points the user provided.
    """
    s = float(max(scale, 0.0))
    if s <= SCALE_REF_NEAR:
        return SCALE_CLOSED_NEAR
    if s >= SCALE_REF_FAR:
        return SCALE_CLOSED_FAR
    if s <= SCALE_REF_MID:
        # interpolate NEAR -> MID
        t = (s - SCALE_REF_NEAR) / (SCALE_REF_MID - SCALE_REF_NEAR)
        return SCALE_CLOSED_NEAR + t * (SCALE_CLOSED_MID - SCALE_CLOSED_NEAR)
    # interpolate MID -> FAR
    t = (s - SCALE_REF_MID) / (SCALE_REF_FAR - SCALE_REF_MID)
    return SCALE_CLOSED_MID + t * (SCALE_CLOSED_FAR - SCALE_CLOSED_MID)

def open_thresh_from_closed(closed_thresh):
    """Derive open threshold from closed threshold, preserving an ambiguous band."""
    raw_open = closed_thresh + OPEN_BAND_OFFSET
    return np.clip(raw_open, MIN_OPEN_THRESH, BASE_OPEN_THRESH)

def get_square_bbox(pts, img_w, img_h, pad_ratio=0.5):
    if not pts: return 0,0,0,0
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    cx = (min(x_coords) + max(x_coords)) // 2
    cy = (min(y_coords) + max(y_coords)) // 2
    max_dim = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
    side_len = int(max_dim * (1.0 + pad_ratio))
    half = side_len // 2
    return max(cx - half, 0), max(cy - half, 0), min(cx + half, img_w), min(cy + half, img_h)

def extract_eye_patch(frame, bbox, size):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1: return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD: return None
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture(CAPTURE_SOURCE)
frame_index = 0

print(f"Starting Capture with Scale Comp... Output: {DATASET_DIR}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    label = None
    
    cur_closed_thresh = BASE_CLOSED_THRESH
    cur_open_thresh = BASE_OPEN_THRESH
    angle_mag = 0
    scale_dev = 0 # Deviation from optimal distance
    current_avg_ear = 0
    face_ratio = OPTIMAL_FACE_RATIO  # default for overlay when no face detected

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        
        # 1. Get Head Pose
        yaw, pitch = get_head_pose_ratios(face)
        angle_mag = abs(yaw) + abs(pitch)
        
        # 2. Get Face Scale (Distance)
        face_ratio = get_face_scale_ratio(face)

        # 3. SCALE CURVE: map face_ratio -> base closed threshold using anchors
        scale_closed = closed_thresh_from_scale(face_ratio)

        # 4. ANGLE CURVE: smoothly boost both thresholds for large pitch/yaw
        ang_f = angle_factor(angle_mag)
        angle_closed_boost = ANGLE_CLOSED_BOOST_MAX * ang_f
        angle_open_drop = ANGLE_OPEN_DROP_MAX * ang_f

        raw_closed = scale_closed + angle_closed_boost
        cur_closed_thresh = min(raw_closed, MAX_CLOSED_THRESH)

        # 5. OPEN THRESHOLD derived from closed plus band, then adjusted by angle
        base_open_from_band = open_thresh_from_closed(cur_closed_thresh)
        raw_open = base_open_from_band - angle_open_drop
        cur_open_thresh = max(raw_open, MIN_OPEN_THRESH)

        # 4. Compute EAR
        left_ear, left_pts = compute_3D_EAR(face, LEFT_EYE, w, h)
        right_ear, right_pts = compute_3D_EAR(face, RIGHT_EYE, w, h)
        current_avg_ear = (left_ear + right_ear) / 2.0

        # 5. Label Logic
        if current_avg_ear > cur_open_thresh:
            if random.random() < OPEN_EYE_SAVE_PROB:
                label = "open"
        elif current_avg_ear < cur_closed_thresh:
            label = "closed"

        # 6. Saving Logic
        if label is not None:
            l_bbox = get_square_bbox(left_pts, w, h, SQUARE_PAD_RATIO)
            r_bbox = get_square_bbox(right_pts, w, h, SQUARE_PAD_RATIO)
            left_patch = extract_eye_patch(frame, l_bbox, PATCH_SIZE)
            right_patch = extract_eye_patch(frame, r_bbox, PATCH_SIZE)

            if left_patch is not None and right_patch is not None:
                base_name = f"{label}_{frame_index}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}"
                
                if SAVE_SEPARATE_EYES:
                    cv2.imwrite(os.path.join(DATASET_DIR, label, f"{base_name}_dyn_L.jpg"), left_patch)
                    cv2.imwrite(os.path.join(DATASET_DIR, label, f"{base_name}_dyn_R.jpg"), right_patch)
                if SAVE_COMBINED:
                    combined = np.hstack([left_patch, right_patch])
                    cv2.imwrite(os.path.join(DATASET_DIR, label, f"{base_name}_dyn_Combined.jpg"), combined)

                # Debug indicator: include head pose and scale in terminal output
                print(
                    f"Saved {label} | yaw: {yaw:.3f} | pitch: {pitch:.3f} | "
                    f"scale: {face_ratio:.3f} | closed_thr: {cur_closed_thresh:.3f} | "
                    f"open_thr: {cur_open_thresh:.3f}"
                )

        # --- VISUALIZATION ---
        if 'l_bbox' in locals(): cv2.rectangle(frame, (l_bbox[0], l_bbox[1]), (l_bbox[2], l_bbox[3]), (255, 255, 0), 1)
        if 'r_bbox' in locals(): cv2.rectangle(frame, (r_bbox[0], r_bbox[1]), (r_bbox[2], r_bbox[3]), (255, 255, 0), 1)

    # UI Overlay
    cv2.putText(frame, f"EAR: {current_avg_ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Closed Thresh: < {cur_closed_thresh:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Distance / Scale Visualization
    dist_color = (0, 255, 0)
    if abs(scale_dev) > 0.1: dist_color = (0, 0, 255) # Red if too far/close

    cv2.putText(
        frame,
        f"Face Scale: {face_ratio:.2f} (Opt: {OPTIMAL_FACE_RATIO})",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        dist_color,
        1,
    )

    frame_index += 1
    cv2.imshow("Dynamic Dataset Gen", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()