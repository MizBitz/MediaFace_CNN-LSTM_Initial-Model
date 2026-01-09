import cv2
import mediapipe as mp
import numpy as np
import os
import random

# ==========================================
#             CONFIGURATION
# ==========================================

# Output Directory
DATASET_DIR = "dataset_dynamic_aligned"

# INPUT: Path to your MEAD dataset folder
MEAD_ROOT_DIR = r"D:\Video Dataset" 

# --- DYNAMIC CURVE CONFIGURATION ---
# 1. Base Values (At mid-range distance, mild pose)
BASE_CLOSED_THRESH = 0.077   
BASE_OPEN_THRESH   = 0.21

# Sensitivity for saving both eyes. 
MAX_Z_DIFF_FOR_BOTH = 0.05

# 2. Maximum Limits (Hard Caps)
MAX_CLOSED_THRESH = 0.115   
MIN_OPEN_THRESH   = 0.180

# --- NEW: CENTER GAP VETO ---
# We use the relaxed 15% threshold to catch squints without blocking real blinks.
CENTER_CLOSURE_THRESH = 0.08

# --- NEW: EAR HARD LOCK ---
# If EAR is super low (below 0.05), it is definitely closed, ignore the Veto.
EAR_HARD_LOCK = 0.05

# 3. ANGLE CURVE CONFIG (Head Pitch)
ANGLE_MAX_MAG = 1.2          
ANGLE_CLOSED_BOOST_MAX = 0.015  
ANGLE_OPEN_DROP_MAX   = 0.020  

# 4. SCALE CURVE CONFIG (Face Distance)
SCALE_REF_NEAR  = 0.04
SCALE_REF_MID   = 0.11
SCALE_REF_FAR   = 0.22
SCALE_CLOSED_NEAR = 0.030
SCALE_CLOSED_MID  = 0.110
SCALE_CLOSED_FAR  = 0.125

# Open threshold band
OPEN_BAND_OFFSET = 0.09   

# 5. SCALE REFERENCE
OPTIMAL_FACE_RATIO = SCALE_REF_MID

# --- Dataset Balancing ---
OPEN_EYE_SAVE_PROB = 0.001 

# --- Image Quality ---
PATCH_SIZE = (64, 64)     
LOG_FILE = "processed_log.txt"

# ==========================================
#           SETUP & HELPERS
# ==========================================

# Create output directories
for state in ["open", "closed"]:
    os.makedirs(os.path.join(DATASET_DIR, state), exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Standard EAR Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# NEW: Center Eyelid Landmarks (Top, Bottom)
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
    """
    Calculates the gap between the precise center of the eyelids
    divided by the eye width.
    """
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

# --- ROBUST ANGLE FACTOR ---
def angle_factor(angle_mag):
    k = 1.5
    # 1. Sanitize Input
    x = max(angle_mag, 0.0)
    
    # 2. Calculate Raw and Max
    raw_val = 1.0 - np.exp(-k * x)
    max_raw = 1.0 - np.exp(-k * ANGLE_MAX_MAG)
    
    # 3. Normalize
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
    raw_open = closed_thresh + OPEN_BAND_OFFSET
    return np.clip(raw_open, MIN_OPEN_THRESH, BASE_OPEN_THRESH)

def get_aligned_eye_crop(frame, landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    cx = int(np.mean(x_vals))
    cy = int(np.mean(y_vals))
    
    sorted_x = sorted(pts, key=lambda k: k[0])
    
    dY = sorted_x[-1][1] - sorted_x[0][1]
    dX = sorted_x[-1][0] - sorted_x[0][0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    eye_width = np.sqrt(dX**2 + dY**2)
    crop_size = int(eye_width * 2.0) 
    if crop_size < 32: crop_size = 32
    
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    
    half = crop_size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    
    crop = rotated_frame[y1:y2, x1:x2]
    return crop

def process_for_save(crop, size):
    if crop.size == 0: return None
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    except:
        return None

# ==========================================
#           MAIN EXECUTION (MEAD)
# ==========================================

processed_videos = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed_videos = set(line.strip() for line in f)

print(f"Resuming... {len(processed_videos)} videos already completed.")

print(f"Scanning MEAD Directory: {MEAD_ROOT_DIR}")
video_files = []
for root, dirs, files in os.walk(MEAD_ROOT_DIR):
    for file in files:
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            full_path = os.path.join(root, file)
            video_files.append(full_path)

if not video_files:
    print("ERROR: No video files found. Check your path!")
    exit()

print(f"Found {len(video_files)} videos. Starting processing...")

for video_idx, video_path in enumerate(video_files):
    if video_path in processed_videos:
        continue 
        
    print(f"[{video_idx+1}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    video_frame_count = 0
    safe_video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        label = None
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            
            # --- 1. DETERMINE ACTIVE EYE & VETO PAIRS ---
            left_dist = face[33].z 
            right_dist = face[362].z
            using_left = (left_dist < right_dist)
            
            if using_left:
                active_eye_indices = LEFT_EYE
                center_pair = LEFT_CENTER_PAIR
                corner_indices = [33, 133] # Left Corners
            else:
                active_eye_indices = RIGHT_EYE
                center_pair = RIGHT_CENTER_PAIR
                corner_indices = [362, 263] # Right Corners

            # --- 2. GET POSE & SCALE ---
            yaw, pitch = get_head_pose_ratios(face)
            angle_mag = abs(yaw) + abs(pitch)
            face_ratio = get_face_scale_ratio(face)

            # --- 3. DYNAMIC THRESHOLDS ---
            scale_closed = closed_thresh_from_scale(face_ratio)
            ang_f = angle_factor(angle_mag)
            
            raw_closed = scale_closed + (ANGLE_CLOSED_BOOST_MAX * ang_f)
            cur_closed_thresh = min(raw_closed, MAX_CLOSED_THRESH)

            base_open = open_thresh_from_closed(cur_closed_thresh)
            cur_open_thresh = max(base_open - (ANGLE_OPEN_DROP_MAX * ang_f), MIN_OPEN_THRESH)

            # --- 4. COMPUTE EAR & CENTER GAP ---
            current_ear, _ = compute_3D_EAR(face, active_eye_indices, w, h)
            center_ratio = get_center_gap_ratio(face, center_pair, corner_indices, w, h)

            # --- 5. LABEL LOGIC WITH VETO ---
            if current_ear > cur_open_thresh:
                if random.random() < OPEN_EYE_SAVE_PROB:
                    label = "open"
            
            elif current_ear < cur_closed_thresh:
                # --- VETO CHECK WITH HARD LOCK ---
                # If EAR is extremely low (<0.05), it is closed. Period.
                # If EAR is just 'low', we check the center gap to ensure it's not a squint.
                if (current_ear < EAR_HARD_LOCK) or (center_ratio < CENTER_CLOSURE_THRESH):
                    label = "closed"
                else:
                    label = None # Squint detected, discard.

            # --- 6. SAVING LOGIC ---
            if label is not None:
                eyes_to_save = []
                z_diff = abs(left_dist - right_dist)
                
                if (left_dist < right_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH):
                    eyes_to_save.append((LEFT_EYE, "L"))
                
                if (right_dist < left_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH):
                    eyes_to_save.append((RIGHT_EYE, "R"))

                for indices, suffix in eyes_to_save:
                    crop_bgr = get_aligned_eye_crop(frame, face, indices, w, h)
                    final_patch = process_for_save(crop_bgr, PATCH_SIZE)

                    if final_patch is not None:
                        filename = f"{label}_MEAD_{safe_video_name}_{video_frame_count}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}_dyn_{suffix}.jpg"
                        save_path = os.path.join(DATASET_DIR, label, filename)
                        cv2.imwrite(save_path, final_patch)

        video_frame_count += 1

    cap.release()
    
    with open(LOG_FILE, "a") as f:
        f.write(video_path + "\n")

cv2.destroyAllWindows()
print("MEAD Processing Complete!")