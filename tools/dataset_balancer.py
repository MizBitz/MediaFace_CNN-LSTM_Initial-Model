import os
import random
import shutil
from tqdm import tqdm  # pip install tqdm (optional, for progress bar)

# ================= CONFIGURATION =================
# 1. Where are the 139k images now?
SOURCE_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset\open"

# 2. Where do you want the selected 10k to go?
DEST_DIR = r"S:\Video Recording\SOP  Test\open"

# 3. How many do you want? (Matching your ~8.8k closed dataset)
TARGET_COUNT = 1000 
# =================================================

def main():
    # 1. Safety Checks
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    # Create destination if it doesn't exist
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")

    # 2. List all files
    print("Scanning source directory...")
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_files = len(all_files)
    
    print(f"Found {total_files} images in source.")

    if total_files < TARGET_COUNT:
        print(f"Warning: Source only has {total_files}, but you asked for {TARGET_COUNT}.")
        print("Copying ALL files...")
        selected_files = all_files
    else:
        print(f"Randomly selecting {TARGET_COUNT} samples...")
        # seed ensures you get the SAME random selection if you run this again later
        random.seed(42) 
        selected_files = random.sample(all_files, TARGET_COUNT)

    # 3. Copy Files
    print("Copying files...")
    for filename in tqdm(selected_files, desc="Copying", unit="img"):
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(DEST_DIR, filename)
        
        shutil.copy2(src_path, dst_path)

    print("\nDone!")
    print(f"New dataset folder '{DEST_DIR}' now contains {len(os.listdir(DEST_DIR))} images.")

if __name__ == "__main__":
    main()