import cv2, os, time
import mediapipe as mp

IMAGE_SIZE = (64, 64)
CONDITIONS = ["bright", "dim", "backlit", "left_pose", "right_pose"]
BASE_OUT = "eval"

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def crop_eye(frame, lm, eye_idx):
    h, w, _ = frame.shape
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in eye_idx]
    x0 = max(min(p[0] for p in pts)-5, 0)
    y0 = max(min(p[1] for p in pts)-5, 0)
    x1 = min(max(p[0] for p in pts)+5, w)
    y1 = min(max(p[1] for p in pts)+5, h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0: return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

def main():
    cap = cv2.VideoCapture(0)
    cond_idx = 0
    os.makedirs(BASE_OUT, exist_ok=True)
    for c in CONDITIONS:
        os.makedirs(os.path.join(BASE_OUT, c, "open"), exist_ok=True)
        os.makedirs(os.path.join(BASE_OUT, c, "closed"), exist_ok=True)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as fm:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            cond = CONDITIONS[cond_idx]
            cv2.putText(frame, f"COND: {cond}  (O=open, C=closed, N=next, Q=quit)",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                eye = crop_eye(frame, lm, LEFT_EYE if lm[33].z < lm[362].z else RIGHT_EYE)
            else:
                eye = None
            cv2.imshow("capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('n'):
                cond_idx = (cond_idx + 1) % len(CONDITIONS)
            if key in (ord('o'), ord('c')) and eye is not None:
                label = "open" if key == ord('o') else "closed"
                fname = f"{int(time.time()*1000)}.jpg"
                cv2.imwrite(os.path.join(BASE_OUT, cond, label, fname), eye)
                print(f"Saved {cond}/{label}/{fname}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()