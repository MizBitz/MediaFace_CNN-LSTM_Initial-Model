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
# UPDATE THIS to the path shown in your screenshot
MEAD_ROOT_DIR = r"D:\Video Dataset" 

# --- DYNAMIC CURVE CONFIGURATION ---
# 1. Base Values (At mid-range distance, mild pose)
BASE_CLOSED_THRESH = 0.077   # UPDATED: Stricter to prevent false positives
BASE_OPEN_THRESH   = 0.21

# Sensitivity for saving both eyes. 
# If head turn is less than this, we save both. If more, we drop the blocked eye.
MAX_Z_DIFF_FOR_BOTH = 0.05

# 2. Maximum Limits (Hard Caps)
MAX_CLOSED_THRESH = 0.115   # UPDATED: Stricter hard cap
MIN_OPEN_THRESH   = 0.180

# 3. ANGLE CURVE CONFIG (Head Pitch)
ANGLE_MAX_MAG = 1.2          
ANGLE_CLOSED_BOOST_MAX = 0.015  
ANGLE_OPEN_DROP_MAX   = 0.020  

# 4. SCALE CURVE CONFIG (Face Distance) - CRITICAL FOR DISTANCE MATH
SCALE_REF_NEAR  = 0.04
SCALE_REF_MID   = 0.11
SCALE_REF_FAR   = 0.22
SCALE_CLOSED_NEAR = 0.030
SCALE_CLOSED_MID  = 0.110
SCALE_CLOSED_FAR  = 0.125

# Open threshold band
OPEN_BAND_OFFSET = 0.09   

# 5. SCALE REFERENCE (for display only)
OPTIMAL_FACE_RATIO = SCALE_REF_MID

# --- Dataset Balancing ---
# Lower this if you get too many open eye samples vs closed
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

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
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
    
    # Yaw
    eye_mid_x = (left_outer.x + right_outer.x) / 2
    face_width = abs(right_outer.x - left_outer.x)
    yaw_ratio = (nose.x - eye_mid_x) / (face_width + 1e-6)
    
    # Pitch
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

def angle_factor(angle_mag):
    k = 1.5
    x = max(angle_mag, 0.0)
    raw = 1.0 - np.exp(-k * x)
    max_raw = 1.0 - np.exp(-k * ANGLE_MAX_MAG)
    if max_raw <= 0: return 0.0
    return np.clip(raw / max_raw, 0.0, 1.0)

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

# --- ALIGNMENT LOGIC (Crucial for Tilted Heads) ---
def get_aligned_eye_crop(frame, landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    cx = int(np.mean(x_vals))
    cy = int(np.mean(y_vals))
    
    # Calculate Angle for Rotation
    sorted_x = sorted(pts, key=lambda k: k[0])
    left_corner = sorted_x[0]
    right_corner = sorted_x[-1]
    
    dY = right_corner[1] - left_corner[1]
    dX = right_corner[0] - left_corner[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Rotate the entire frame around the eye center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Dynamic crop size
    eye_width = np.sqrt(dX**2 + dY**2)
    crop_size = int(eye_width * 2.0) 
    if crop_size < 32: crop_size = 32
    
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    
    # Crop from Rotated Frame
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

# 1. Load the Progress Log (Resume Capability)
processed_videos = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed_videos = set(line.strip() for line in f)

print(f"Resuming... {len(processed_videos)} videos already completed.")

# 2. Gather all video files
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
    # --- RESUME CHECK ---
    if video_path in processed_videos:
        continue # Skip already done videos
        
    print(f"[{video_idx+1}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    video_frame_count = 0
    
    # Safe name for file saving
    safe_video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break # End of video
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        label = None
        
        # Default Calculation Variables
        cur_closed_thresh = BASE_CLOSED_THRESH
        cur_open_thresh = BASE_OPEN_THRESH
        current_ear = 0
        face_ratio = OPTIMAL_FACE_RATIO  
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            
            # --- 1. DETERMINE ACTIVE EYE & DEPTH ---
            left_dist = face[33].z 
            right_dist = face[362].z
            using_left = (left_dist < right_dist)
            
            if using_left:
                active_eye_indices = LEFT_EYE
            else:
                active_eye_indices = RIGHT_EYE

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

            # --- 4. COMPUTE EAR (Active Eye Only) ---
            current_ear, _ = compute_3D_EAR(face, active_eye_indices, w, h)

            # --- 5. LABEL LOGIC ---
            if current_ear > cur_open_thresh:
                if random.random() < OPEN_EYE_SAVE_PROB:
                    label = "open"
            elif current_ear < cur_closed_thresh:
                label = "closed"

            # --- 6. SAVING LOGIC (Safe Zone + Alignment) ---
            if label is not None:
                eyes_to_save = []
                
                # Check Head Turn Severity
                z_diff = abs(left_dist - right_dist)
                
                # Save Left? (Yes if closer OR if head turn is small)
                if (left_dist < right_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH):
                    eyes_to_save.append((LEFT_EYE, "L"))
                
                # Save Right? (Yes if closer OR if head turn is small)
                if (right_dist < left_dist) or (z_diff < MAX_Z_DIFF_FOR_BOTH):
                    eyes_to_save.append((RIGHT_EYE, "R"))

                for indices, suffix in eyes_to_save:
                    # ALIGNMENT HAPPENS HERE
                    crop_bgr = get_aligned_eye_crop(frame, face, indices, w, h)
                    
                    # GRAYSCALE & RESIZE
                    final_patch = process_for_save(crop_bgr, PATCH_SIZE)

                    if final_patch is not None:
                        # Construct unique filename
                        # filename = f"{label}_{safe_video_name}_{video_frame_count}_{suffix}.jpg"
                        filename = f"{label}_MEAD_{safe_video_name}_{video_frame_count}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}_dyn_{suffix}.jpg"
                        
                        save_path = os.path.join(DATASET_DIR, label, filename)
                        cv2.imwrite(save_path, final_patch)

        video_frame_count += 1

    cap.release()
    
    # --- MARK VIDEO AS DONE ---
    with open(LOG_FILE, "a") as f:
        f.write(video_path + "\n")

cv2.destroyAllWindows()
print("MEAD Processing Complete!")