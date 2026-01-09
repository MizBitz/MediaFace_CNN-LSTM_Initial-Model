import cv2
import os
import shutil

# ================= CONFIGURATION =================
# ONLY point this to your OPEN folder. 
# Do not use on 'closed' folder.
TARGET_FOLDER = r"S:\VSCode Projects\MediaFace\dataset_uncleaned\open" 
TRASH_FOLDER = r"S:\VSCode Projects\MediaFace\dataset_uncleaned\blurry images\messy"
# =================================================

def clean_non_eyes():
    # 1. Load the "Eye with Glasses" Detector
    # This is built-in to OpenCV and handles glasses frames better than the standard one.
    cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    
    if eye_cascade.empty():
        print("Error: Could not find haar cascade xml file.")
        return

    print(f"Scanning for non-eyes in: {TARGET_FOLDER}")
    os.makedirs(TRASH_FOLDER, exist_ok=True)
    
    files = [f for f in os.listdir(TARGET_FOLDER) if f.lower().endswith('.jpg')]
    moved_count = 0
    kept_count = 0

    for file in files:
        file_path = os.path.join(TARGET_FOLDER, file)
        
        # Load Image
        img = cv2.imread(file_path)
        if img is None: continue
        
        # 2. Pre-processing
        # We MUST upscale the image temporarily. 
        # Haar Cascades struggle with tiny 64x64 images. 
        # Upscaling to 128x128 makes the eye features "big enough" to detect.
        large_img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)

        # 3. Detect Eyes
        # scaleFactor=1.05: Scans very thoroughly (slow but accurate)
        # minNeighbors=3: Requires decent confidence. Lower to 2 if it deletes too many good ones.
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))

        # 4. Decision Logic
        if len(eyes) == 0:
            # NO EYE DETECTED -> Likely Garbage (Glasses frame, cheek, etc.)
            print(f"REJECT: {file} (No eye structure found)")
            try:
                shutil.move(file_path, os.path.join(TRASH_FOLDER, file))
                moved_count += 1
            except Exception as e:
                print(f"Error moving: {e}")
        else:
            kept_count += 1

    print(f"\nScan Complete.")
    print(f"Kept:  {kept_count} (Valid Eyes)")
    print(f"Moved: {moved_count} (Suspicious/geometric artifacts)")
    print(f"Check '{TRASH_FOLDER}' to verify.")

if __name__ == "__main__":
    clean_non_eyes()