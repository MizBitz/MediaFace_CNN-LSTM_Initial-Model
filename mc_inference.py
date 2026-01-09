import cv2
import numpy as np
import onnxruntime as ort
import time
import mediapipe as mp
import json
import win32com.client # Direct Windows SAPI access (more stable than pyttsx3)
import threading
import queue
import pythoncom # Required for stable TTS in threads on Windows
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
CHAR_PAUSE_THRESHOLD = 1.0  
WORD_PAUSE_THRESHOLD = 2.0  
# Word pause only starts after being idle for this long (no committed letters)
WORD_PAUSE_GRACE = 2.0
MIN_OPEN_STABILITY = 0.1 # seconds; debounce time to ignore blink glitches

# Head-gesture controls
# Uses a simple yaw ratio from FaceMesh landmarks.
# Note: depending on camera mirroring, you may need to flip RIGHT/LEFT.
HEAD_BACKSPACE_ENABLED = True
HEAD_BACKSPACE_DIRECTION = "RIGHT"  # "RIGHT" or "LEFT"
HEAD_BACKSPACE_YAW_THRESHOLD = 0.35  # higher = more turn required
HEAD_BACKSPACE_YAW_RESET = 0.20      # hysteresis reset threshold (must return below this)
HEAD_BACKSPACE_HOLD_SEC = 0.25       # must hold the turn for this long
HEAD_BACKSPACE_COOLDOWN_SEC = 0.75   # minimum time between backspaces

# ==========================================
#           TEXT TO SPEECH SETUP
# ==========================================
speech_queue = queue.Queue()

def speech_worker():
    """Persistent background thread that handles all speech requests"""
    # CRITICAL: Initialize COM for this thread on Windows
    pythoncom.CoInitialize()
    
    try:
        # Initialize the native Windows voice engine directly
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
    except Exception as e:
        print(f"SAPI Initialization Error: {e}")
        return
    
    while True:
        text = speech_queue.get()
        if text is None: break 
        
        try:
            # Direct Speak call is much faster and more stable in threads
            speaker.Speak(text)
        except Exception as e:
            print(f"Speech error: {e}")
        
        speech_queue.task_done()
    
    pythoncom.CoUninitialize()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text(text):
    """Adds text to the speech queue"""
    if text and text != "?":
        speech_queue.put(text)

# ==========================================
#           LOAD MODELS
# ==========================================
print("Loading models...")

# Load BOTH maps for switching
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

# State for Mode Switching
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
        # Button area: (800, 10) to (980, 50)
        if 800 <= x <= 980 and 10 <= y <= 50:
            toggle_mode()

# Load models with flexible provider selection
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
LEFT_EYE = [33, 160, 158, 133, 153, 144]

NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

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
    # Shape to (1, T, 1) as expected by LSTM
    arr = np.array([[min(d, 2.0) for d in raw_durations]], dtype=np.float32)[:, :, None]
    feed = {"input": arr}
    if "lengths" in lstm_inputs:
        feed["lengths"] = np.array([arr.shape[1]], dtype=np.int64)
    
    logits = lstm_sess.run(None, feed)[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])
    return idx_to_label.get(pred_idx, "?")

def get_eye_crop(frame, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
    x_vals, y_vals = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(x_vals) + max(x_vals)) // 2, (min(y_vals) + max(y_vals)) // 2
    max_dim = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
    side = int(max_dim * 2.0) # Slightly larger crop for better features
    half = side // 2
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = min(cx + half, w), min(cy + half, h)
    return frame[y1:y2, x1:x2]

def get_yaw_ratio(landmarks):
    """Approx head yaw estimate: nose horizontal offset from eye-mid, normalized by face width."""
    nose = landmarks[NOSE_TIP]
    left_outer = landmarks[LEFT_EYE_OUTER]
    right_outer = landmarks[RIGHT_EYE_OUTER]

    eye_mid_x = (left_outer.x + right_outer.x) / 2.0
    face_width = abs(right_outer.x - left_outer.x)
    if face_width < 1e-6:
        return 0.0
    return (nose.x - eye_mid_x) / face_width

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
    last_token_time = last_open_time  # last time we committed a letter/word
    current_blink_sequence = []
    decoded_history = [] # List of strings instead of single string

    # Word/sentence assembly (CHAR mode builds words from letters)
    current_word = ""
    transcript_words = []  # list[str]

    # Head-turn backspace state
    head_turn_start = None
    head_backspace_armed = True
    head_backspace_cooldown_until = 0.0

    def handle_backspace(now_ts: float):
        """Delete last character (CHAR mode) or last word token (WORD mode)."""
        nonlocal current_word, transcript_words, last_token_time

        if current_mode == "CHAR":
            if current_word:
                current_word = current_word[:-1]
            elif transcript_words:
                last = transcript_words[-1]
                if len(last) <= 1:
                    transcript_words.pop()
                else:
                    transcript_words[-1] = last[:-1]
        else:
            if transcript_words:
                transcript_words.pop()
            elif current_word:
                current_word = ""

        # Prevent immediate auto-commit after editing
        last_token_time = now_ts
    
    print("System Ready! SAPI TTS & Timers Enabled.")
    
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
        cnn_val = 1 # Debug raw value
        now = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Head-turn backspace gesture (runs even if eye crop fails)
            if HEAD_BACKSPACE_ENABLED:
                yaw = get_yaw_ratio(landmarks)
                # Choose direction. If mirrored, swap RIGHT/LEFT or negate yaw.
                if HEAD_BACKSPACE_DIRECTION.upper() == "RIGHT":
                    turn_val = yaw
                else:
                    turn_val = -yaw

                if turn_val > HEAD_BACKSPACE_YAW_THRESHOLD and now >= head_backspace_cooldown_until and head_backspace_armed:
                    if head_turn_start is None:
                        head_turn_start = now
                    elif (now - head_turn_start) >= HEAD_BACKSPACE_HOLD_SEC:
                        handle_backspace(now)
                        head_backspace_cooldown_until = now + HEAD_BACKSPACE_COOLDOWN_SEC
                        head_backspace_armed = False
                        head_turn_start = None
                else:
                    # Not currently above threshold; clear timer.
                    head_turn_start = None

                # Re-arm only once the head returns near center (hysteresis)
                if not head_backspace_armed and turn_val < HEAD_BACKSPACE_YAW_RESET:
                    head_backspace_armed = True

            crop = get_eye_crop(frame, landmarks, w, h)
            
            if crop.size != 0:
                # 1. Run CNN
                cnn_out = ort_session.run(None, {cnn_input_name: preprocess_cnn(crop)})
                logits = cnn_out[0]

                # train_cnn.py exports 2-class logits with shape (B, 2) (typically B=1)
                # Class indices are assumed: 0=Closed, 1=Open
                if isinstance(logits, list):
                    logits = np.asarray(logits)
                logits = np.asarray(logits)

                if logits.ndim == 2 and logits.shape[0] >= 1 and logits.shape[1] >= 2:
                    cnn_val = int(np.argmax(logits[0, :2]))
                elif logits.ndim == 1 and logits.shape[0] >= 2:
                    cnn_val = int(np.argmax(logits[:2]))
                else:
                    # Unexpected output shape; default to OPEN to avoid false blinks
                    cnn_val = 1
                
                if cnn_val == 0: 
                    eye_state = "CLOSED"
                    # Treat a detected closure as "activity" so we don't trigger CHAR_PAUSE while
                    # a blink/hold is still in progress.
                    last_open_time = now
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
                            # Blink finished
                            duration = potential_open_start - closed_start_time
                            is_closed = False
                            potential_open_start = None
                            if duration > 0.05:
                                current_blink_sequence.append(duration)
                                print(f"Blink recorded: {duration:.2f}s")
                            last_open_time = now

        # 3. Prediction Timer Logic
        time_since_last_blink = now - last_open_time
        
        # Trigger Prediction (commit a token after a "letter pause")
        if len(current_blink_sequence) > 0 and time_since_last_blink > CHAR_PAUSE_THRESHOLD:
            predicted = predict_letter(current_blink_sequence)

            if predicted and predicted != "?":
                if current_mode == "CHAR":
                    # Build words from letters
                    current_word += predicted
                    decoded_history.append(predicted)
                    print(f"Letter: {predicted}")
                else:
                    # WORD mode: treat prediction as a full word token
                    transcript_words.append(predicted)
                    decoded_history.append(predicted)
                    speak_text(predicted)
                    print(f"Word: {predicted}")

                last_token_time = now

            current_blink_sequence = []
            last_open_time = now  # reset pause timer after committing a token

        # Word boundary:
        # Only start counting a word-pause after we've been idle (no committed letters) for WORD_PAUSE_GRACE seconds.
        time_since_last_token = now - last_token_time
        if (
            current_mode == "CHAR"
            and current_word
            and len(current_blink_sequence) == 0
            and time_since_last_token > (WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD)
        ):
            transcript_words.append(current_word)
            speak_text(current_word)  # speak whole word
            print(f"Word committed: {current_word}")
            current_word = ""
            last_token_time = now

        # Human-readable transcript
        transcript_text = " ".join(transcript_words + ([current_word] if current_word else []))

        # --- UI DRAWING ---
        # Create Canvas
        canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 240 # Light gray background

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
        
        # Calculate aspect-ratio preserving resize
        h_frame, w_frame = frame.shape[:2]
        scale = min(VIDEO_WIDTH / w_frame, VIDEO_HEIGHT / h_frame)
        new_w = int(w_frame * scale)
        new_h = int(h_frame * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Draw overlays on the video feed
        color = (0, 0, 255) if eye_state == "CLOSED" else (0, 255, 0)
        cv2.putText(frame_resized, f"Eye: {eye_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Center the video in the box
        y_offset = vy + (VIDEO_HEIGHT - new_h) // 2
        x_offset = vx + (VIDEO_WIDTH - new_w) // 2
        
        # Draw black background for video box
        cv2.rectangle(canvas, (vx, vy), (vx+VIDEO_WIDTH, vy+VIDEO_HEIGHT), (0, 0, 0), -1)
        
        # Place video on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
        cv2.rectangle(canvas, (vx, vy), (vx+VIDEO_WIDTH, vy+VIDEO_HEIGHT), (0, 0, 0), 2)
        cv2.putText(canvas, "Live Video Feed", (vx + 10, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 2. Translation Field (Current Sequence)
        tx, ty = TRANS_FIELD_POS
        tw, th = TRANS_FIELD_SIZE
        cv2.rectangle(canvas, (tx, ty), (tx+tw, ty+th), (255, 255, 255), -1)
        cv2.rectangle(canvas, (tx, ty), (tx+tw, ty+th), (0, 0, 0), 1)
        # Label inside the box, smaller
        cv2.putText(canvas, "Translation Field", (tx + 5, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Visualize current blink sequence as dots/dashes
        seq_str = ""
        for dur in current_blink_sequence:
            if dur < 0.5: seq_str += "."
            else: seq_str += "-"

        # Show assembled text + current blink pattern
        wrapped = textwrap.wrap(transcript_text, width=34)
        if wrapped:
            cv2.putText(canvas, wrapped[-1], (tx + 10, ty + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            cv2.putText(canvas, "(waiting)", (tx + 10, ty + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        cv2.putText(canvas, seq_str, (tx + 10, ty + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)

        # 3. Translation History
        hx, hy = HISTORY_POS
        hw, hh = HISTORY_SIZE
        cv2.rectangle(canvas, (hx, hy), (hx+hw, hy+hh), (230, 230, 230), -1)
        cv2.rectangle(canvas, (hx, hy), (hx+hw, hy+hh), (0, 0, 0), 1)
        cv2.putText(canvas, "Translation History", (hx + 10, hy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(canvas, (hx, hy+40), (hx+hw, hy+40), (0,0,0), 1)

        # Draw the decoded history (last 15 entries)
        y_offset = hy + 70
        # Show most recent words (and partial current word)
        visible_history = (transcript_words + ([current_word] if current_word else []))[-15:]
        for item in visible_history:
            cv2.putText(canvas, item, (hx+10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_offset += 35

        cv2.imshow("LSTM Morse Decoder with TTS", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        # Backspace support: Backspace key is commonly 8 (sometimes 127). 'b' is a fallback.
        if key in (8, 127) or key == ord('b'):
            handle_backspace(now)
            continue
        if key == ord('c'):
            cap.release()
            current_cam_idx += 1
            if current_cam_idx > 3: current_cam_idx = 0 # Cycle 0-3
            cap = cv2.VideoCapture(current_cam_idx)
            print(f"Switching to camera {current_cam_idx}...")

    # Exit
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()