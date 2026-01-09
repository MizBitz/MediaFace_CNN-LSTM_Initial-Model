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
import winsound
import os
from datetime import datetime

# ==========================================
#           CONFIGURATION
# ==========================================
CNN_MODEL_PATH = r"S:\VSCode Projects\MediaFace\eye_state_mobilenet.onnx"
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

# Header buttons
AUDIO_BTN_RECT = (600, 10, 780, 50)  # x1,y1,x2,y2
MODE_BTN_RECT = (800, 10, 980, 50)

# Timing Thresholds
CHAR_PAUSE_THRESHOLD = 1.0  
WORD_PAUSE_THRESHOLD = 2.0  
# Word pause only starts after being idle for this long (no committed letters)
WORD_PAUSE_GRACE = 2.0
SENTENCE_SPEAK_BLINK_SEC = 2.0  # long blink gesture: speak the full current transcript
BATCH_COMMIT_BLINK_SEC = 2.0    # long blink gesture: decode buffered Morse
MIN_OPEN_STABILITY = 0.1 # seconds; debounce time to ignore blink glitches

# Evaluation / logging (SOP4 support)
EVAL_PROMPT_MODE = False           # If True: show prompts, score attempts, log metrics
EVAL_FORCE_CHAR_MODE = True        # If True: lock to CHAR mode during prompt eval
EVAL_PROMPTS_PATH = "eval_prompts.txt"  # one prompt per line

LOGGING_ENABLED = True
LOG_DIR = "session_logs"

# Head-gesture controls
# Uses a simple yaw ratio from FaceMesh landmarks.
# Note: depending on camera mirroring, you may need to flip RIGHT/LEFT.
HEAD_BACKSPACE_ENABLED = True
HEAD_BACKSPACE_DIRECTION = "RIGHT"  # "RIGHT" or "LEFT"
HEAD_BACKSPACE_YAW_THRESHOLD = 0.25  # higher = more turn required
HEAD_BACKSPACE_YAW_RESET = 0.20      # hysteresis reset threshold (must return below this)
HEAD_BACKSPACE_HOLD_SEC = 0.15       # must hold the turn for this long
HEAD_BACKSPACE_COOLDOWN_SEC = 1   # minimum time between backspaces

# After a backspace (keyboard or head gesture), ignore blink inputs briefly.
POST_BACKSPACE_INPUT_BLOCK_SEC = 0.75

# CNN output handling
# The exported MobileNet model emits a single logit: logit > 0 -> open.
# Adjust this threshold if you later want a stricter closed detection.
OPEN_PROB_THRESHOLD = 0.5

# ==========================================
#           TEXT TO SPEECH SETUP
# ==========================================
speech_queue = queue.Queue()

# Audio output mode
# - TTS: speak words via SAPI
# - MUTE: no audio
# - BEEP: play short/long beeps matching Morse input (dot/dash)
current_audio_mode = "TTS"  # "TTS" | "MUTE" | "BEEP"

# Morse beep settings (Windows)
BEEP_FREQ_HZ = 880
DOT_BEEP_MS = 120
DASH_BEEP_MS = 360
BEEP_GAP_MS = 60

beep_queue = queue.Queue()

def beep_worker():
    """Background thread that plays queued beeps so the main loop stays real-time."""
    while True:
        item = beep_queue.get()
        if item is None:
            break
        try:
            freq_hz, dur_ms = item
            winsound.Beep(int(freq_hz), int(dur_ms))
            if BEEP_GAP_MS > 0:
                time.sleep(BEEP_GAP_MS / 1000.0)
        except Exception as e:
            print(f"Beep error: {e}")
        finally:
            beep_queue.task_done()

beep_thread = threading.Thread(target=beep_worker, daemon=True)
beep_thread.start()

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
    if current_audio_mode != "TTS":
        return
    if text and text != "?":
        speech_queue.put(text)

def play_morse_beep(duration_sec: float):
    """Enqueue a dot/dash beep based on the same threshold used for rendering."""
    if current_audio_mode != "BEEP":
        return
    is_dot = duration_sec < 0.5
    beep_queue.put((BEEP_FREQ_HZ, DOT_BEEP_MS if is_dot else DASH_BEEP_MS))


def _normalize_for_eval(text: str) -> str:
    # Normalize whitespace/case to reduce trivial mismatches
    return " ".join((text or "").strip().lower().split())


def _levenshtein_distance(a: str, b: str) -> int:
    # Classic DP; good enough for short prompts.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _cer(pred: str, truth: str) -> float:
    truth_n = _normalize_for_eval(truth)
    pred_n = _normalize_for_eval(pred)
    if not truth_n:
        return 0.0 if not pred_n else 1.0
    dist = _levenshtein_distance(pred_n, truth_n)
    return dist / max(1, len(truth_n))


def _wer(pred: str, truth: str) -> float:
    truth_words = _normalize_for_eval(truth).split()
    pred_words = _normalize_for_eval(pred).split()
    if not truth_words:
        return 0.0 if not pred_words else 1.0

    # Word-level Levenshtein
    prev = list(range(len(pred_words) + 1))
    for i, tw in enumerate(truth_words, start=1):
        curr = [i]
        for j, pw in enumerate(pred_words, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if tw == pw else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1] / max(1, len(truth_words))


MORSE_TABLE = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}


def decode_morse_tokens(tokens: list[str]) -> str:
    """Decode a list of morse letter tokens into text.

    Tokens are typically like ['.-', '.-.', '---'].
    Use '/' token to indicate a word boundary.
    """
    out: list[str] = []
    for tok in tokens:
        t = (tok or "").strip()
        if not t:
            continue
        if t in ("/", "|"):
            # Word boundary
            if out and out[-1] != " ":
                out.append(" ")
            continue
        out.append(MORSE_TABLE.get(t, "?"))

    # Collapse any repeated spaces
    return "".join(out).replace("  ", " ").strip()

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
current_mode = "WORD"  # WORD | CHAR | BUFFER
idx_to_label = idx_to_word

def toggle_audio_mode():
    global current_audio_mode
    if current_audio_mode == "TTS":
        current_audio_mode = "MUTE"
    elif current_audio_mode == "MUTE":
        current_audio_mode = "BEEP"
    else:
        current_audio_mode = "TTS"
    print(f"Switched to {current_audio_mode} audio")

def toggle_mode():
    global current_mode, idx_to_label
    if current_mode == "WORD":
        current_mode = "CHAR"
        idx_to_label = idx_to_char
    elif current_mode == "CHAR":
        current_mode = "BUFFER"
        idx_to_label = idx_to_char
    else:
        current_mode = "WORD"
        idx_to_label = idx_to_word
    print(f"Switched to {current_mode} mode")

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ax1, ay1, ax2, ay2 = AUDIO_BTN_RECT
        mx1, my1, mx2, my2 = MODE_BTN_RECT

        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            toggle_audio_mode()
            return

        if mx1 <= x <= mx2 and my1 <= y <= my2:
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
    last_blink_duration = 0.0
    decoded_history = [] # List of strings instead of single string

    # Word/sentence assembly (CHAR mode builds words from letters)
    current_word = ""
    transcript_words = []  # list[str]

    # Buffered Morse mode state
    buffered_tokens: list[str] = []  # completed Morse tokens (".-", "--" ... or "/")
    current_morse_token = ""       # in-progress token while blinking in BUFFER mode

    input_block_until = 0.0

    last_sentence_speak_time = 0.0

    # --- Session logging / evaluation state ---
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_events_f = None
    session_summary_path = None

    if LOGGING_ENABLED or EVAL_PROMPT_MODE:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_events_path = os.path.join(LOG_DIR, f"mc_session_{session_id}.jsonl")
        session_summary_path = os.path.join(LOG_DIR, f"mc_session_{session_id}_summary.json")
        log_events_f = open(log_events_path, "a", encoding="utf-8")

        summary = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "CNN_MODEL_PATH": CNN_MODEL_PATH,
                "LSTM_MODEL_PATH": LSTM_MODEL_PATH,
                "CHAR_PAUSE_THRESHOLD": CHAR_PAUSE_THRESHOLD,
                "WORD_PAUSE_THRESHOLD": WORD_PAUSE_THRESHOLD,
                "WORD_PAUSE_GRACE": WORD_PAUSE_GRACE,
                "SENTENCE_SPEAK_BLINK_SEC": SENTENCE_SPEAK_BLINK_SEC,
                "MIN_OPEN_STABILITY": MIN_OPEN_STABILITY,
                "OPEN_PROB_THRESHOLD": OPEN_PROB_THRESHOLD,
            },
            "eval": {
                "enabled": EVAL_PROMPT_MODE,
                "force_char_mode": EVAL_FORCE_CHAR_MODE,
                "prompts_path": EVAL_PROMPTS_PATH,
                "attempts": [],
            },
        }
    else:
        summary = None

    eval_prompts = []
    eval_prompt_idx = 0
    if EVAL_PROMPT_MODE:
        try:
            if os.path.exists(EVAL_PROMPTS_PATH):
                with open(EVAL_PROMPTS_PATH, "r", encoding="utf-8") as f:
                    eval_prompts = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Prompt load error: {e}")
            eval_prompts = []

        # Fallback prompts if file missing/empty
        if not eval_prompts:
            eval_prompts = [
                "hello world",
                "this is a test",
                "openai",
                "morse decoder",
            ]

        if summary is not None:
            summary["eval"]["prompts"] = eval_prompts

        if EVAL_FORCE_CHAR_MODE:
            # Lock to CHAR mode for consistent scoring across subjects.
            global current_mode, idx_to_label
            current_mode = "CHAR"
            idx_to_label = idx_to_char

    # Head-turn backspace state
    head_turn_start = None
    head_backspace_armed = True
    head_backspace_cooldown_until = 0.0

    def handle_backspace(now_ts: float):
        """Delete last entry: BUFFER edits Morse, otherwise edits transcript."""
        nonlocal current_word, transcript_words, last_token_time, input_block_until, buffered_tokens, current_morse_token, current_blink_sequence

        if current_mode == "BUFFER":
            if current_morse_token:
                current_morse_token = current_morse_token[:-1]
                current_blink_sequence = []
                last_token_time = now_ts
                input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)
                return
            if buffered_tokens:
                buffered_tokens.pop()
                current_blink_sequence = []
                last_token_time = now_ts
                input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)
                return
            # Fall through to transcript editing if buffer is empty (post-decode edits)

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

        # Prevent accidental dot/dash inputs immediately after a backspace gesture.
        input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)
    
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
                logits = np.asarray(cnn_out[0])

                # Support both 2-class logits (legacy) and single-logit (current MobileNet export).
                flat = logits.reshape(-1)
                if flat.size >= 2:
                    # 2-class path: 0=Closed, 1=Open
                    cnn_val = int(np.argmax(flat[:2]))
                elif flat.size == 1:
                    # 1-logit path: sigmoid(logit) = P(open)
                    logit = float(flat[0])
                    open_prob = 1.0 / (1.0 + np.exp(-logit))
                    cnn_val = 1 if open_prob >= OPEN_PROB_THRESHOLD else 0
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
                            last_blink_duration = duration
                            is_closed = False
                            potential_open_start = None
                            if now >= input_block_until:
                                # Long-blink actions
                                if current_mode == "BUFFER" and duration >= BATCH_COMMIT_BLINK_SEC:
                                    # Finalize any in-progress Morse token
                                    if current_morse_token:
                                        buffered_tokens.append(current_morse_token)
                                        current_morse_token = ""

                                    decoded_text = decode_morse_tokens(buffered_tokens)
                                    buffered_tokens = []
                                    current_blink_sequence = []

                                    if decoded_text:
                                        transcript_words.append(decoded_text)
                                        decoded_history.append(decoded_text)
                                        speak_text(decoded_text)
                                        print(f"Buffered decode: {decoded_text}")
                                        last_token_time = now

                                        if log_events_f is not None:
                                            log_events_f.write(json.dumps({
                                                "ts": now,
                                                "type": "buffer_decode",
                                                "decoded": decoded_text,
                                                "mode": current_mode,
                                            }, ensure_ascii=False) + "\n")
                                            log_events_f.flush()
                                # Long-blink command gesture: speak full sentence (current transcript)
                                elif duration >= SENTENCE_SPEAK_BLINK_SEC and (now - last_sentence_speak_time) >= 0.5:
                                    transcript_text_now = " ".join(transcript_words + ([current_word] if current_word else []))
                                    speak_text(transcript_text_now)
                                    print(f"Speak sentence: {transcript_text_now}")
                                    last_sentence_speak_time = now

                                    if log_events_f is not None:
                                        log_events_f.write(json.dumps({
                                            "ts": now,
                                            "type": "gesture_speak_sentence",
                                            "blink_duration_sec": duration,
                                            "mode": current_mode,
                                            "transcript": transcript_text_now,
                                            "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                                        }, ensure_ascii=False) + "\n")
                                        log_events_f.flush()

                                    # If running eval prompts: treat long blink as "submit attempt".
                                    if EVAL_PROMPT_MODE and eval_prompts:
                                        target = eval_prompts[eval_prompt_idx]
                                        pred = transcript_text_now
                                        attempt = {
                                            "prompt_index": eval_prompt_idx,
                                            "prompt": target,
                                            "predicted": pred,
                                            "cer": _cer(pred, target),
                                            "wer": _wer(pred, target),
                                            "submitted_at_ts": now,
                                        }
                                        if summary is not None:
                                            summary["eval"]["attempts"].append(attempt)

                                        # Advance to next prompt and clear transcript so attempts are separated.
                                        eval_prompt_idx = (eval_prompt_idx + 1) % len(eval_prompts)
                                        transcript_words = []
                                        current_word = ""
                                        last_token_time = now
                                elif duration > 0.05:
                                    current_blink_sequence.append(duration)
                                    print(f"Blink recorded: {duration:.2f}s")
                                    play_morse_beep(duration)

                                    if current_mode == "BUFFER":
                                        # Build the Morse token without decoding yet
                                        if duration < 0.5:
                                            current_morse_token += "."
                                        else:
                                            current_morse_token += "-"

                                    if log_events_f is not None:
                                        log_events_f.write(json.dumps({
                                            "ts": now,
                                            "type": "blink_recorded",
                                            "blink_duration_sec": duration,
                                        }, ensure_ascii=False) + "\n")
                                        log_events_f.flush()
                            else:
                                # Discard any blink that completes during the post-backspace lockout.
                                current_blink_sequence = []
                            last_open_time = now

        # 3. Prediction Timer Logic
        time_since_last_blink = now - last_open_time
        
        # Trigger Prediction (commit a token after a "letter pause")
        if now >= input_block_until and len(current_blink_sequence) > 0 and time_since_last_blink > CHAR_PAUSE_THRESHOLD:
            if current_mode == "BUFFER":
                if current_morse_token:
                    buffered_tokens.append(current_morse_token)
                    current_morse_token = ""
                    last_token_time = now
            else:
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

                    if log_events_f is not None:
                        transcript_text_now = " ".join(transcript_words + ([current_word] if current_word else []))
                        log_events_f.write(json.dumps({
                            "ts": now,
                            "type": "token_committed",
                            "mode": current_mode,
                            "token": predicted,
                            "blink_sequence_sec": list(current_blink_sequence),
                            "transcript": transcript_text_now,
                            "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                        }, ensure_ascii=False) + "\n")
                        log_events_f.flush()

                    last_token_time = now

            current_blink_sequence = []
            last_open_time = now  # reset pause timer after committing a token

        # Word boundary:
        # Only start counting a word-pause after we've been idle (no committed letters) for WORD_PAUSE_GRACE seconds.
        time_since_last_token = now - last_token_time
        if (
            now >= input_block_until
            and current_mode == "CHAR"
            and current_word
            and len(current_blink_sequence) == 0
            and time_since_last_token > (WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD)
        ):
            transcript_words.append(current_word)
            speak_text(current_word)  # speak whole word
            print(f"Word committed: {current_word}")

            if log_events_f is not None:
                transcript_text_now = " ".join(transcript_words)
                log_events_f.write(json.dumps({
                    "ts": now,
                    "type": "word_boundary_commit",
                    "word": current_word,
                    "mode": current_mode,
                    "transcript": transcript_text_now,
                    "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                }, ensure_ascii=False) + "\n")
                log_events_f.flush()

            current_word = ""
            last_token_time = now

        # Word boundary hint for BUFFER mode: insert a "/" separator after a long pause
        if (
            now >= input_block_until
            and current_mode == "BUFFER"
            and not current_morse_token
            and len(buffered_tokens) > 0
            and len(current_blink_sequence) == 0
            and time_since_last_token > (WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD)
        ):
            buffered_tokens.append("/")
            last_token_time = now

        # Human-readable transcript
        transcript_text = " ".join(transcript_words + ([current_word] if current_word else []))

        # --- UI DRAWING ---
        # Create Canvas
        canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 240 # Light gray background

        # Header
        cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 60), (200, 200, 200), -1)
        header_text = "GROUP 2 WIP"
        if EVAL_PROMPT_MODE and eval_prompts:
            header_text = f"PROMPT: {eval_prompts[eval_prompt_idx]}"
        cv2.putText(canvas, header_text[:40], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Audio Button
        audio_color = (200, 200, 200)
        if current_audio_mode == "TTS":
            audio_color = (100, 200, 100)
        elif current_audio_mode == "MUTE":
            audio_color = (120, 120, 120)
        else:
            audio_color = (200, 160, 80)

        ax1, ay1, ax2, ay2 = AUDIO_BTN_RECT
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), audio_color, -1)
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (0, 0, 0), 1)
        cv2.putText(canvas, f"AUDIO: {current_audio_mode}", (ax1 + 10, ay2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Mode Button
        btn_color = (100, 200, 100) if current_mode == "WORD" else (100, 100, 200)
        mx1, my1, mx2, my2 = MODE_BTN_RECT
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), btn_color, -1)
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (0, 0, 0), 1)
        cv2.putText(canvas, f"MODE: {current_mode}", (mx1 + 10, my2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
        blink_text = "Blink: --" if last_blink_duration <= 0 else f"Blink: {last_blink_duration:.2f}s"
        cv2.putText(frame_resized, blink_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        
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
        label_text = "Translation Field"
        if current_mode == "BUFFER":
            label_text = "Buffer Mode (hold 2s to decode)"
        cv2.putText(canvas, label_text, (tx + 5, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Visualize current blink sequence as dots/dashes
        if current_mode == "BUFFER":
            seq_parts = buffered_tokens.copy()
            if current_morse_token:
                seq_parts.append(current_morse_token)
            seq_str = " ".join(seq_parts) if seq_parts else "(Decoding...)"
        else:
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

        # Optional: skip to next prompt (eval mode)
        if key == ord('n') and EVAL_PROMPT_MODE and eval_prompts:
            eval_prompt_idx = (eval_prompt_idx + 1) % len(eval_prompts)
            transcript_words = []
            current_word = ""
            last_token_time = now

    # Exit
    beep_queue.put(None)
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

    if summary is not None and session_summary_path is not None:
        summary["ended_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            with open(session_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write session summary: {e}")

    if log_events_f is not None:
        try:
            log_events_f.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()