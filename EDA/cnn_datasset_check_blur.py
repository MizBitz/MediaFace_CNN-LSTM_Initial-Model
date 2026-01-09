import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
#           CONFIGURATION
# ==========================================
DATASET_DIR = r"S:\VSCode Projects\MediaFace\dataset_dynamic_aligned"  # Your dataset folder
EXTENSIONS = ["*.jpg", "*.png", "*.jpeg"]

def calculate_sharpness_scores(root_dir):
    sharpness_scores = []
    image_paths = []
    
    print(f"Scanning {root_dir}...")
    
    # Recursively find all images
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    
    print(f"Found {len(files)} images. Calculating Laplacian variance...")
    
    for filepath in files:
        # Read image
        img = cv2.imread(filepath)
        if img is None: continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- THE MAGIC METRIC ---
        # Calculate Variance of Laplacian
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        sharpness_scores.append(score)
        image_paths.append(filepath)

    return sharpness_scores, image_paths

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found.")
        return

    scores, paths = calculate_sharpness_scores(DATASET_DIR)
    
    if not scores:
        print("No images found.")
        return

# ==========================================
    #           VISUALIZATION
    # ==========================================
    plt.figure(figsize=(14, 6)) # Made it wider to fit the numbers
    
    # 1. Calculate the range for your ticks
    max_val = int(np.max(scores))
    # Create a list of numbers from 0 to max_val, stepping by 50
    # We add 51 to ensure the last number is included
    x_ticks = np.arange(0, max_val + 50, 50) 
    
    # 2. Plot Histogram
    # 'bins' controls the bars. We set it to match your 50-step desire roughly,
    # or you can use a fixed number like 100 for high detail.
    plt.hist(scores, bins=100, color='purple', alpha=0.7, edgecolor='black')
    
    # 3. Apply the custom X-axis ticks
    plt.xticks(x_ticks, rotation=45) # Rotate text so they don't overlap
    
    plt.title("Image Sharpness Distribution (Laplacian Variance)")
    plt.xlabel("Sharpness Score (Variance)")
    plt.ylabel("Number of Images")
    
    # Add a reference line for "Blurry" (Common threshold is 45)
    plt.axvline(x=45, color='red', linestyle='dashed', linewidth=2, label='Blur Threshold (45)')
    plt.legend()
    
    plt.grid(axis='both', alpha=0.3) # Grid on both X and Y helps read values
    plt.tight_layout() # Fixes layout if numbers get cut off
    plt.show()

    # ==========================================
    #           STATISTICS
    # ==========================================
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"\n--- Statistics ---")
    print(f"Total Images: {len(scores)}")
    print(f"Average Sharpness: {avg_score:.2f}")
    print(f"Min Sharpness: {min_score:.2f} (Blurriest)")
    print(f"Max Sharpness: {max_score:.2f} (Sharpest)")
    
    # Count images below threshold
    blurry_count = sum(1 for s in scores if s < 45)
    print(f"\nPotential Blurry Images (< 45): {blurry_count} ({blurry_count/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    main()