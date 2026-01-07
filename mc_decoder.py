import cv2
import numpy as np
import onnxruntime as ort
import time
import mediapipe as mp
from statistics import median
import json

MODEL_PATH = "eye_state_cnn.onnx"
IMAGE_SIZE = (64, 64)

# Default time-based Morse thresholds (used if calibration is skipped)
DOT_DURATION_MAX = 0.5   # seconds; shorter = dot
DASH_DURATION_MIN = 0.6  # seconds; longer = dash
CHAR_PAUSE_MIN = 3.0     # open gap between blinks to end a character
WORD_PAUSE_MIN = 9.0     # open gap between characters to mark word boundary
MIN_OPEN_STABILITY = 0.1 # seconds; debounce time to ignore blink glitches

# Data collection settings
COLLECTION_MODE = True           # Set True when collecting training data
TARGET_LABEL = "A"               # Change this manually when recording different letters
SEQ_LOG_PATH = "morse_sequences.jsonl" 

# Calibration targets
CAL_DOT_COUNT = 8
CAL_DASH_COUNT = 8

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Left eye indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]

def load_session():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def get_square_bbox(pts, img_w, img_h, pad_ratio=0.8):
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    cx = (min(x_coords) + max(x_coords)) // 2
    cy = (min(y_coords) + max(y_coords)) // 2
    max_dim = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
    side_len = int(max_dim * (1.0 + pad_ratio))
    half = side_len // 2
    return max(cx - half, 0), max(cy - half, 0), min(cx + half, img_w), min(cy + half, img_h)

def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5
    return np.expand_dims(np.expand_dims(norm, axis=0), axis=0)

def run_calibration(cap, session, input_name):
    print("Calibration: Blink quick for dots, then long for dashes. Press 'q' to skip.")
    phases = [
        {"name": "DOT", "target": CAL_DOT_COUNT, "durations": []},
        {"name": "DASH", "target": CAL_DASH_COUNT, "durations": []},
    ]
    phase_idx = 0
    is_closed = False
    eye_closed_start = None

    while phase_idx < len(phases):
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            pts = [(int(face[i].x * w), int(face[i].y * h)) for i in LEFT_EYE]
            x1, y1, x2, y2 = get_square_bbox(pts, w, h)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size != 0:
                input_tensor = preprocess(crop)
                outputs = session.run(None, {input_name: input_tensor})
                prediction = int(np.argmax(outputs[0]))
                now = time.time()

                if prediction == 0: # CLOSED
                    if not is_closed:
                        eye_closed_start = now
                        is_closed = True
                else: # OPEN
                    if is_closed and eye_closed_start:
                        duration = now - eye_closed_start
                        is_closed = False
                        phases[phase_idx]["durations"].append(duration)
                        print(f"Cal {phases[phase_idx]['name']}: {duration:.2f}s")

        cv2.putText(frame, f"Cal {phases[phase_idx]['name']} {len(phases[phase_idx]['durations'])}/{phases[phase_idx]['target']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyWindow("Calibration")
            return None
        
        if len(phases[phase_idx]["durations"]) >= phases[phase_idx]["target"]:
            phase_idx += 1
            is_closed = False
            eye_closed_start = None

    cv2.destroyWindow("Calibration")
    dot_med = median(phases[0]["durations"])
    dash_med = median(phases[1]["durations"])
    if dash_med <= dot_med: return None

    return {
        "DOT_DURATION_MAX": dot_med * 1.2,
        "DASH_DURATION_MIN": (dot_med + dash_med) * 0.5,
        "CHAR_PAUSE_MIN": dot_med * 8.0,
        "WORD_PAUSE_MIN": dot_med * 12.0
    }

def main():
    session, input_name = load_session()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    cal = run_calibration(cap, session, input_name)
    dot_max = cal["DOT_DURATION_MAX"] if cal else DOT_DURATION_MAX
    dash_min = cal["DASH_DURATION_MIN"] if cal else DASH_DURATION_MIN
    char_pause = cal["CHAR_PAUSE_MIN"] if cal else CHAR_PAUSE_MIN
    word_pause = cal["WORD_PAUSE_MIN"] if cal else WORD_PAUSE_MIN

    is_closed = False
    eye_closed_start = None
    potential_open_start = None
    last_blink_end = time.time()
    current_symbol = ""
    events_for_lstm = [] 

    print(f"Decoder ready. Recording label: {TARGET_LABEL}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        state_label = "OPEN"
        now = time.time()

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            pts = [(int(face[i].x * w), int(face[i].y * h)) for i in LEFT_EYE]
            x1, y1, x2, y2 = get_square_bbox(pts, w, h)
            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:
                input_tensor = preprocess(crop)
                outputs = session.run(None, {input_name: input_tensor})
                prediction = int(np.argmax(outputs[0]))

                if prediction == 0: # CLOSED
                    state_label = "CLOSED"
                    potential_open_start = None
                    if not is_closed:
                        eye_closed_start = now
                        is_closed = True
                else: # OPEN
                    state_label = "OPEN"
                    if is_closed and eye_closed_start:
                        if potential_open_start is None:
                            potential_open_start = now
                        
                        if (now - potential_open_start) > MIN_OPEN_STABILITY:
                            duration = potential_open_start - eye_closed_start
                            is_closed = False
                            last_blink_end = potential_open_start
                            potential_open_start = None
                            
                            # Interpret symbol for UI
                            symbol = "." if duration < dot_max else "-" if duration > dash_min else "?"
                            current_symbol += symbol
                            
                            # Log raw event for LSTM
                            events_for_lstm.append({
                                "duration_s": duration,
                                "timestamp": last_blink_end
                            })
                            print(f"Blink: {duration:.2f}s -> {symbol}")

        # Check for letter complete (Gap)
        gap_since_last_blink = now - last_blink_end

        if gap_since_last_blink > char_pause and current_symbol:
            print(f"Letter finished: {current_symbol}")
            
            if COLLECTION_MODE:
                # NEW: Extract just the float durations
                raw_durations = [e["duration_s"] for e in events_for_lstm]
                
                sample = {
                    "raw_durations": raw_durations,  # <--- CRITICAL FOR LSTM
                    "morse_seq": current_symbol,     # Keep for reference
                    "label": TARGET_LABEL
                }
                
                try:
                    with open(SEQ_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample) + "\n")
                    print(f"Saved sample for {TARGET_LABEL} with {len(raw_durations)} blinks.")
                except Exception as e:
                    print(f"Save error: {e}")

            # RESET buffers
            current_symbol = ""
            events_for_lstm = [] 
            
            if gap_since_last_blink > word_pause:
                print("<word boundary>")

        # UI Overlay
        color = (0, 0, 255) if state_label == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"State: {state_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Seq: {current_symbol}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Blink Decoder", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()