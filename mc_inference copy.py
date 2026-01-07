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

# ==========================================
#           CONFIGURATION
# ==========================================
CNN_MODEL_PATH = "eye_state_cnn.onnx"
LSTM_MODEL_PATH = "blink_lstm.onnx"
LABEL_MAP_PATH = "lstm_word_map.json"
IMAGE_SIZE = (64, 64)

# Timing Thresholds
CHAR_PAUSE_THRESHOLD = 2.0  
WORD_PAUSE_THRESHOLD = 5.0  
MIN_OPEN_STABILITY = 0.1 # seconds; debounce time to ignore blink glitches

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

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

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

# ==========================================
#           MAIN LOOP
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    is_closed = False
    closed_start_time = 0
    potential_open_start = None
    last_open_time = time.time()
    current_blink_sequence = []
    decoded_sentence = ""
    
    print("System Ready! SAPI TTS & Timers Enabled.")

    while cap.isOpened():
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
            crop = get_eye_crop(frame, landmarks, w, h)
            
            if crop.size != 0:
                # 1. Run CNN
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
        
        # Trigger Prediction
        if len(current_blink_sequence) > 0 and time_since_last_blink > CHAR_PAUSE_THRESHOLD:
            predicted_char = predict_letter(current_blink_sequence)
            if predicted_char:
                decoded_sentence += predicted_char
                speak_text(predicted_char)
                print(f"Result: {predicted_char}")
            
            current_blink_sequence = []
            last_open_time = now # Reset timer to avoid immediate space

        # Space detection
        if time_since_last_blink > WORD_PAUSE_THRESHOLD and decoded_sentence and not decoded_sentence.endswith(" "):
            decoded_sentence += " "
            print("Space added")

        # --- UI DRAWING ---
        # Darken the top area for stats
        cv2.rectangle(frame, (0,0), (w, 110), (0,0,0), -1)
        
        # Eye State
        color = (0, 0, 255) if eye_state == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"Eye: {eye_state} ({cnn_val})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Buffer (Blinks captured so far)
        buffer_str = " ".join([f"{x:.1f}" for x in current_blink_sequence])
        cv2.putText(frame, f"Blinks: [{buffer_str}]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Timer (Countdown to prediction)
        if len(current_blink_sequence) > 0:
            remaining = max(0, CHAR_PAUSE_THRESHOLD - time_since_last_blink)
            timer_color = (0, 255, 255) if remaining > 0.5 else (0, 165, 255)
            cv2.putText(frame, f"Predicting in: {remaining:.1f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)

        # Output Text (The final sentence)
        cv2.putText(frame, f"Sentence: {decoded_sentence}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

        cv2.imshow("LSTM Morse Decoder with TTS", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Exit
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()