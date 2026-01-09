import cv2
import os
import shutil

# ================= CONFIGURATION =================
# Point this to your CLOSED folder (images that should show closed eyes).
TARGET_FOLDER = r"S:\VSCode Projects\MediaFace\dataset_dynamic_aligned\closed"
# Where to move suspected open-eye frames or obvious artifacts.
TRASH_FOLDER = r"S:\VSCode Projects\MediaFace\dataset_dynamic_aligned\blurry images\closed_trash_opens"
# =================================================


def clean_closed_folder():
    # Haar cascade that tends to fire on visible eye structures; useful to spot accidental open-eye frames.
    cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)

    if eye_cascade.empty():
        print("Error: Could not find haar cascade xml file.")
        return

    print(f"Scanning for open-eye hits in: {TARGET_FOLDER}")
    os.makedirs(TRASH_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(TARGET_FOLDER) if f.lower().endswith(".jpg")]
    moved_count = 0
    kept_count = 0

    for file in files:
        file_path = os.path.join(TARGET_FOLDER, file)

        img = cv2.imread(file_path)
        if img is None:
            print(f"REJECT: {file} (Unreadable image)")
            try:
                shutil.move(file_path, os.path.join(TRASH_FOLDER, file))
                moved_count += 1
            except Exception as e:
                print(f"Error moving: {e}")
            continue

        # Upscale small 64x64 crops to help the cascade detect eye structures.
        large_img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)

        # If the cascade detects eye-like structures, treat it as an accidental open-eye frame.
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))

        if len(eyes) > 0:
            print(f"REJECT (looks open): {file} (Detections: {len(eyes)})")
            try:
                shutil.move(file_path, os.path.join(TRASH_FOLDER, file))
                moved_count += 1
            except Exception as e:
                print(f"Error moving: {e}")
        else:
            kept_count += 1

    print("\nScan Complete.")
    print(f"Kept:  {kept_count} (Likely closed eyes)")
    print(f"Moved: {moved_count} (Suspected open eyes / artifacts)")
    print(f"Check '{TRASH_FOLDER}' to verify.")


if __name__ == "__main__":
    clean_closed_folder()