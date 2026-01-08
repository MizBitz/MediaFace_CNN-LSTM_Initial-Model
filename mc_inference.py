import cv2
import numpy as np
import onnxruntime as ort
import time
import mediapipe as mp
import json
import win32com.client 
import threading
import queue
import pythoncom 
import textwrap

# ==========================================
#           CONFIGURATION
# ==========================================
CNN_MODEL_PATH = "eye_state_cnn.onnx"
LSTM_MODEL_PATH = "blink_lstm.onnx"
LABEL_MAP_PATH = "lstm_word_map.json"
IMAGE_SIZE = (64, 64)

# UI Layout Dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_POS = (20, 80)
TRANS_FIELD_POS = (20, 580)
TRANS_FIELD_SIZE = (640, 100)
HISTORY_POS = (680, 80)
HISTORY_SIZE = (300, 600)

# Timing Thresholds
CHAR_PAUSE_THRESHOLD = 2.0  
WORD_PAUSE_THRESHOLD = 5.0  
MIN_OPEN_STABILITY = 0.1 

# ==========================================
#           TEXT TO SPEECH SETUP
# ==========================================
speech_queue = queue.Queue()

def speech_worker():
    """Persistent background thread that handles all speech requests"""
    pythoncom.CoInitialize()
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
    except Exception as e:
        print(f"SAPI Initialization Error: {e}")
        return
    
    while True:
        text = speech_queue.get()
        if text is None: break 
        try:
            speaker.Speak(text)
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()
    pythoncom.CoUninitialize()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text(text):
    if text and text != "?":
        speech_queue.put(text)

# ==========================================
#           LOAD MODELS
# ==========================================
print("Loading models...")

try:
    with open("lstm_word_map.json", "r") as f:
        word_map = json.load(f)
        idx_to_word = {v: k for k, v in word_map.items()}
except Exception as e:
    print(f"Error loading word map: {e}")
    idx_to_word = {}

try:
    with open("lstm_label_map.json", "r") as f:
        char_map = json.load(f)
        idx_to_char = {v: k for k, v in char_map.items()}
except Exception as e:
    print(f"Error loading label map: {e}")
    idx_to_char = {}

current_mode = "WORD"
idx_to_label = idx_to_word

def toggle_mode():
    global current_mode, idx_to_label
    if current_mode == "WORD":
        current_mode = "CHAR"
        idx_to_label = idx_to_char
    else:
        current_mode = "WORD"
        idx_to_label = idx_to_word
    print(f"Switched to {current_mode} mode")

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 800 <= x <= 980 and 10 <= y <= 50:
            toggle_mode()

def create_session(path):
    try:
        return ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

lstm_sess = create_session(LSTM_MODEL_PATH)
lstm_inputs = {i.name: i for i in lstm_sess.get_inputs()}

ort_session = create_session(CNN_MODEL_PATH)
cnn_input_name = ort_session.get_inputs()[0].name

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Define Both Eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
#           HELPER FUNCTIONS
# ==========================================
def preprocess_cnn(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5
    return np.expand_dims(np.expand_dims(norm, axis=0), axis=0)

def predict_letter(raw_durations):
    if not raw_durations: return ""
    arr = np.array([[min(d, 2.0) for d in raw_durations]], dtype=np.float32)[:, :, None]
    feed = {"input": arr}
    if "lengths" in lstm_inputs:
        feed["lengths"] = np.array([arr.shape[1]], dtype=np.int64)
    
    logits = lstm_sess.run(None, feed)[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])
    return idx_to_label.get(pred_idx, "?")

def get_aligned_eye_crop(frame, landmarks, eye_indices, w, h):
    """
    Rotates the frame to align the eye horizontally before cropping.
    Increases detection accuracy when head is tilted.
    """
    # Get all points for the eye
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    # Calculate Center of the eye
    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    cx = int(np.mean(x_vals))
    cy = int(np.mean(y_vals))
    
    # Sort points by x-coord to find corners
    sorted_x = sorted(pts, key=lambda k: k[0])
    left_corner = sorted_x[0]
    right_corner = sorted_x[-1]
    
    # Calculate Angle for Rotation (Head Tilt)
    dY = right_corner[1] - left_corner[1]
    dX = right_corner[0] - left_corner[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Get Rotation Matrix (rotate around the eye center)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # Determine dynamic crop size based on eye width
    eye_width = np.sqrt(dX**2 + dY**2)
    crop_size = int(eye_width * 2.0) # 2x multiplier ensures we get the whole eye + skin
    if crop_size < 32: crop_size = 32 # Minimum safety size
    
    # Perform Affine Warp (Rotate the whole image around the eye center)
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    
    # Crop from the ROTATED frame
    half = crop_size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    
    crop = rotated_frame[y1:y2, x1:x2]
    
    # Check if crop is valid (sometimes rotation pushes it off edge)
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.array([])
        
    return crop

# ==========================================
#           MAIN LOOP
# ==========================================
def main():
    current_cam_idx = 0
    cap = cv2.VideoCapture(current_cam_idx)
    is_closed = False
    closed_start_time = 0
    potential_open_start = None
    last_open_time = time.time()
    current_blink_sequence = []
    decoded_history = [] 
    
    print("System Ready! Alignment & Switching Enabled.")
    
    cv2.namedWindow("LSTM Morse Decoder with TTS")
    cv2.setMouseCallback("LSTM Morse Decoder with TTS", mouse_callback)

    while True:
        if not cap.isOpened():
            print(f"Camera {current_cam_idx} failed. Resetting to 0.")
            current_cam_idx = 0
            cap = cv2.VideoCapture(current_cam_idx)
            if not cap.isOpened(): break

        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        eye_state = "OPEN"
        active_eye_name = "LEFT"
        cnn_val = 1 
        now = time.time()

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark
            
            # --- DYNAMIC EYE SWITCHING LOGIC ---
            # Compare Z-depth of inner eye corners
            # 33 is Left Inner, 362 is Right Inner
            left_dist = landmarks[33].z 
            right_dist = landmarks[362].z
            
            # Pick the eye closer to the camera (smaller Z)
            if left_dist < right_dist:
                active_eye_indices = LEFT_EYE
                active_eye_name = "LEFT"
            else:
                active_eye_indices = RIGHT_EYE
                active_eye_name = "RIGHT"

            # Use new Alignment Function
            crop = get_aligned_eye_crop(frame, landmarks, active_eye_indices, w, h)
            
            if crop.size != 0:
                # Run CNN
                cnn_out = ort_session.run(None, {cnn_input_name: preprocess_cnn(crop)})
                cnn_val = np.argmax(cnn_out[0]) # 0=Closed, 1=Open
                
                if cnn_val == 0: 
                    eye_state = "CLOSED"
                    potential_open_start = None
                    if not is_closed:
                        is_closed = True
                        closed_start_time = now
                else:
                    eye_state = "OPEN"
                    if is_closed:
                        if potential_open_start is None:
                            potential_open_start = now
                        
                        if (now - potential_open_start) > MIN_OPEN_STABILITY:
                            duration = potential_open_start - closed_start_time
                            is_closed = False
                            potential_open_start = None
                            if duration > 0.05:
                                current_blink_sequence.append(duration)
                                print(f"Blink recorded: {duration:.2f}s")
                            last_open_time = now

        # Prediction Timer Logic
        time_since_last_blink = now - last_open_time
        
        if len(current_blink_sequence) > 0 and time_since_last_blink > CHAR_PAUSE_THRESHOLD:
            predicted_char = predict_letter(current_blink_sequence)
            if predicted_char:
                decoded_history.append(predicted_char)
                speak_text(predicted_char)
                print(f"Result: {predicted_char}")
            
            current_blink_sequence = []
            last_open_time = now

        # --- UI DRAWING ---
        canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 240 

        # Header
        cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 60), (200, 200, 200), -1)
        cv2.putText(canvas, "GROUP 2 WIP", (WINDOW_WIDTH//2 - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Mode Button
        btn_color = (100, 200, 100) if current_mode == "WORD" else (100, 100, 200)
        cv2.rectangle(canvas, (800, 10), (980, 50), btn_color, -1)
        cv2.rectangle(canvas, (800, 10), (980, 50), (0, 0, 0), 1)
        cv2.putText(canvas, f"MODE: {current_mode}", (810, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 1. Live Video Feed
        vx, vy = VIDEO_POS
        
        h_frame, w_frame = frame.shape[:2]
        scale = min(VIDEO_WIDTH / w_frame, VIDEO_HEIGHT / h_frame)
        new_w = int(w_frame * scale)
        new_h = int(h_frame * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Draw overlays on the video feed
        color = (0, 0, 255) if eye_state == "CLOSED" else (0, 255, 0)
        cv2.putText(frame_resized, f"Eye: {eye_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame_resized, f"Active: {active_eye_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        y_offset = vy + (VIDEO_HEIGHT - new_h) // 2
        x_offset = vx + (VIDEO_WIDTH - new_w) // 2
        
        cv2.rectangle(canvas, (vx, vy), (vx+VIDEO_WIDTH, vy+VIDEO_HEIGHT), (0, 0, 0), -1)
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
        cv2.rectangle(canvas, (vx, vy), (vx+VIDEO_WIDTH, vy+VIDEO_HEIGHT), (0, 0, 0), 2)
        cv2.putText(canvas, "Live Video Feed", (vx + 10, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 2. Translation Field
        tx, ty = TRANS_FIELD_POS
        tw, th = TRANS_FIELD_SIZE
        cv2.rectangle(canvas, (tx, ty), (tx+tw, ty+th), (255, 255, 255), -1)
        cv2.rectangle(canvas, (tx, ty), (tx+tw, ty+th), (0, 0, 0), 1)
        cv2.putText(canvas, "Translation Field", (tx + 5, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        seq_str = ""
        for dur in current_blink_sequence:
            if dur < 0.5: seq_str += "."
            else: seq_str += "-"
        
        cv2.putText(canvas, seq_str, (tx+20, ty+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # 3. Translation History
        hx, hy = HISTORY_POS
        hw, hh = HISTORY_SIZE
        cv2.rectangle(canvas, (hx, hy), (hx+hw, hy+hh), (230, 230, 230), -1)
        cv2.rectangle(canvas, (hx, hy), (hx+hw, hy+hh), (0, 0, 0), 1)
        cv2.putText(canvas, "Translation History", (hx + 10, hy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(canvas, (hx, hy+40), (hx+hw, hy+40), (0,0,0), 1)

        y_off = hy + 70
        visible_history = decoded_history[-15:]
        for item in visible_history:
            cv2.putText(canvas, item, (hx+10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_off += 35

        cv2.imshow("LSTM Morse Decoder with TTS", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            cap.release()
            current_cam_idx += 1
            if current_cam_idx > 3: current_cam_idx = 0 
            cap = cv2.VideoCapture(current_cam_idx)
            print(f"Switching to camera {current_cam_idx}...")

    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()