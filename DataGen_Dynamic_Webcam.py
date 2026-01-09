import cv2
import mediapipe as mp
import numpy as np
import os
import random

# ==========================================
#             CONFIGURATION
# ==========================================

# 1. Output Directory
DATASET_DIR = "dataset_dynamic_aligned"

# 2. Capture Source (0 = Webcam)
CAPTURE_SOURCE = 0

# --- DYNAMIC CURVE CONFIGURATION ---
BASE_CLOSED_THRESH = 0.08   
BASE_OPEN_THRESH   = 0.21

# Sensitivity for saving both eyes. 
MAX_Z_DIFF_FOR_BOTH = 0.05

# Maximum Limits (Hard Caps)
MAX_CLOSED_THRESH = 0.120
MIN_OPEN_THRESH   = 0.180

# --- CENTER GAP VETO (UPDATED) ---
# We relaxed this from 0.05 to 0.15. 
# This means the center gap must be smaller than 15% of the eye width.
CENTER_CLOSURE_THRESH = 0.08 

# --- SAFETY OVERRIDE ---
# If EAR is lower than this, we force "Closed" regardless of the veto.
EAR_HARD_LOCK = 0.05

BLUR_THRESHOLD = 35.0

# Angle Curve Config
ANGLE_MAX_MAG = 1.2          
ANGLE_CLOSED_BOOST_MAX = 0.015  
ANGLE_OPEN_DROP_MAX   = 0.020  

# Scale Curve Config
SCALE_REF_NEAR  = 0.04
SCALE_REF_MID   = 0.11
SCALE_REF_FAR   = 0.22
SCALE_CLOSED_NEAR = 0.030
SCALE_CLOSED_MID  = 0.110
SCALE_CLOSED_FAR  = 0.125
OPEN_BAND_OFFSET = 0.09   

# Balancing
OPEN_EYE_SAVE_PROB = 0.15 
PATCH_SIZE = (64, 64)     
OPTIMAL_FACE_RATIO = SCALE_REF_MID

# ==========================================
#           SETUP & HELPERS
# ==========================================

# Create directories
for state in ["open", "closed"]:
    os.makedirs(os.path.join(DATASET_DIR, state), exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Center Eyelid Pairs (for Veto Check)
LEFT_CENTER_PAIR = [159, 145]
RIGHT_CENTER_PAIR = [386, 374]

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

def get_center_gap_ratio(landmarks, center_pair, corner_indices, w, h):
    """Calculates the gap between the exact center of the eyelids."""
    p_top = get_3d_point(landmarks[center_pair[0]], w, h)
    p_bot = get_3d_point(landmarks[center_pair[1]], w, h)
    
    p_left = get_3d_point(landmarks[corner_indices[0]], w, h)
    p_right = get_3d_point(landmarks[corner_indices[1]], w, h)
    
    center_dist = np.linalg.norm(p_top - p_bot)
    width_dist = np.linalg.norm(p_left - p_right)
    
    if width_dist == 0: return 1.0
    return center_dist / width_dist

def get_head_pose_ratios(landmarks):
    nose = landmarks[NOSE_TIP]
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    eye_mid_x = (left_outer.x + right_outer.x) / 2
    face_width = abs(right_outer.x - left_outer.x)
    yaw_ratio = (nose.x - eye_mid_x) / (face_width + 1e-6)
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    face_height = abs(chin.y - forehead.y)
    face_mid_y = (forehead.y + chin.y) / 2
    pitch_ratio = (nose.y - face_mid_y) / (face_height + 1e-6)
    return yaw_ratio, pitch_ratio

def get_face_scale_ratio(landmarks):
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    return abs(right_outer.x - left_outer.x)

# --- FIXED ANGLE FACTOR ---
def angle_factor(angle_mag):
    k = 1.5
    x = max(angle_mag, 0.0)
    raw_val = 1.0 - np.exp(-k * x)
    max_raw = 1.0 - np.exp(-k * ANGLE_MAX_MAG)
    if max_raw <= 0: return 0.0
    return np.clip(raw_val / max_raw, 0.0, 1.0)

def closed_thresh_from_scale(scale):
    s = float(max(scale, 0.0))
    if s <= SCALE_REF_NEAR: return SCALE_CLOSED_NEAR
    if s >= SCALE_REF_FAR: return SCALE_CLOSED_FAR
    if s <= SCALE_REF_MID:
        t = (s - SCALE_REF_NEAR) / (SCALE_REF_MID - SCALE_REF_NEAR)
        return SCALE_CLOSED_NEAR + t * (SCALE_CLOSED_MID - SCALE_CLOSED_NEAR)
    t = (s - SCALE_REF_MID) / (SCALE_REF_FAR - SCALE_REF_MID)
    return SCALE_CLOSED_MID + t * (SCALE_CLOSED_FAR - SCALE_CLOSED_MID)

def open_thresh_from_closed(closed_thresh):
    return np.clip(closed_thresh + OPEN_BAND_OFFSET, MIN_OPEN_THRESH, BASE_OPEN_THRESH)

def get_aligned_eye_crop(frame, landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    cx = int(np.mean(x_vals))
    cy = int(np.mean(y_vals))
    
    sorted_x = sorted(pts, key=lambda k: k[0])
    angle = np.degrees(np.arctan2(sorted_x[-1][1] - sorted_x[0][1], sorted_x[-1][0] - sorted_x[0][0]))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    eye_width = np.sqrt((sorted_x[-1][0] - sorted_x[0][0])**2 + (sorted_x[-1][1] - sorted_x[0][1])**2)
    crop_size = max(int(eye_width * 2.0), 32)
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    half = crop_size // 2
    return rotated_frame[max(cy-half, 0):min(cy+half, h), max(cx-half, 0):min(cx+half, w)]

def process_for_save(crop, size):
    """
    Grayscales, Checks for Blur, and Resizes.
    Returns None if the image is too blurry.
    """
    if crop.size == 0: return None
    
    try:
        # 1. Convert to Gray
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. BLUR CHECK (Inserted Here)
        # We calculate sharpness BEFORE resizing, as resizing can hide blur.
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < BLUR_THRESHOLD:
            # Optional: Print to console so you know it's happening
            # print(f"Rejected Blur: {blur_score:.1f}") 
            return None

        # 3. Resize and Return
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        
    except Exception as e:
        print(f"Error processing patch: {e}")
        return None

# ==========================================
#           MAIN CAPTURE LOOP
# ==========================================

cap = cv2.VideoCapture(CAPTURE_SOURCE)
frame_index = 0

print(f"Starting Webcam Capture... Output: {DATASET_DIR}")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    label = None
    status_text = "Searching..."
    status_color = (200, 200, 200)

    cur_closed_thresh = BASE_CLOSED_THRESH
    cur_open_thresh = BASE_OPEN_THRESH
    current_ear = 0
    center_ratio = 1.0
    active_eye_name = "NONE"

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        
        # 1. Active Eye
        left_dist, right_dist = face[33].z, face[362].z
        using_left = (left_dist < right_dist)
        
        if using_left:
            active_eye_indices = LEFT_EYE
            active_eye_name = "LEFT"
            center_pair = LEFT_CENTER_PAIR
            corner_indices = [33, 133]
        else:
            active_eye_indices = RIGHT_EYE
            active_eye_name = "RIGHT"
            center_pair = RIGHT_CENTER_PAIR
            corner_indices = [362, 263]

        # 2. Thresholds
        yaw, pitch = get_head_pose_ratios(face)
        face_ratio = get_face_scale_ratio(face)
        ang_f = angle_factor(abs(yaw) + abs(pitch))
        
        cur_closed_thresh = min(closed_thresh_from_scale(face_ratio) + (ANGLE_CLOSED_BOOST_MAX * ang_f), MAX_CLOSED_THRESH)
        cur_open_thresh = max(open_thresh_from_closed(cur_closed_thresh) - (ANGLE_OPEN_DROP_MAX * ang_f), MIN_OPEN_THRESH)

        # 3. Compute Metrics
        current_ear, eye_pts = compute_3D_EAR(face, active_eye_indices, w, h)
        center_ratio = get_center_gap_ratio(face, center_pair, corner_indices, w, h)

        # 4. Classification
        if current_ear > cur_open_thresh:
            if random.random() < OPEN_EYE_SAVE_PROB:
                label = "open"
                status_text = "SAVING OPEN"
                status_color = (0, 255, 0)
        
        elif current_ear < cur_closed_thresh:
            # --- VETO CHECK WITH HARD LOCK ---
            # If EAR is super low (EAR_HARD_LOCK), we save it no matter what (it's definitely closed).
            # If EAR is normal-low, we apply the Veto to stop squints.
            if (current_ear < EAR_HARD_LOCK) or (center_ratio < CENTER_CLOSURE_THRESH):
                label = "closed"
                status_text = "SAVING CLOSED"
                status_color = (0, 0, 255)
            else:
                label = None
                status_text = f"VETOED (Squint: {center_ratio:.2f})"
                status_color = (0, 165, 255)

        # 5. Save Logic
        if label is not None:
            eyes_to_save = []
            z_diff = abs(left_dist - right_dist)
            if (left_dist < right_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH): eyes_to_save.append((LEFT_EYE, "L"))
            if (right_dist < left_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH): eyes_to_save.append((RIGHT_EYE, "R"))

            for indices, suffix in eyes_to_save:
                crop_bgr = get_aligned_eye_crop(frame, face, indices, w, h)
                final_patch = process_for_save(crop_bgr, PATCH_SIZE)
                if final_patch is not None:
                    # UPDATED NAMING
                    filename = f"{label}_Webcam_{frame_index}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}_dyn_{suffix}.jpg"
                    cv2.imwrite(os.path.join(DATASET_DIR, label, filename), final_patch)

        for pt in eye_pts: cv2.circle(frame, pt, 1, (0, 255, 0), -1)

    # UI Overlay
    cv2.putText(frame, f"Active: {active_eye_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Visual check for Center Gap
    cv2.putText(frame, f"Center: {center_ratio:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.putText(frame, status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    frame_index += 1
    cv2.imshow("Webcam DataGen", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()