import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np

# ================= CONFIGURATION =================
DEFAULT_IMAGE_SIZE = (64, 64)

# Define the exact augmentation values here so they are consistent everywhere
AUGMENT_CONFIG = {
    'degrees': 10,
    'translate': (0.05, 0.05),
    'scale': (0.95, 1.05),
    'shear': 5,
    'blur_sigma': (0.1, 1.0),
    'blur_prob': 0.15,
    # --- NEW: LIGHTING ---
    'brightness': 0.3, # Variation factor (0.3 means +/- 30% brightness)
    'contrast': 0.3    # Variation factor (0.3 means +/- 30% contrast)
}
# =================================================

def get_train_transform(image_size=DEFAULT_IMAGE_SIZE):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        
        # --- NEW: LIGHTING AUGMENTATION ---
        # Simulates dark rooms, bright windows, and high/low contrast webcams
        transforms.ColorJitter(
            brightness=AUGMENT_CONFIG['brightness'], 
            contrast=AUGMENT_CONFIG['contrast']
        ),

        # 1. Texture Augmentation (Blur)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=AUGMENT_CONFIG['blur_sigma'])
        ], p=AUGMENT_CONFIG['blur_prob']),
        
        # 2. Geometric Augmentation (Position/Shape)
        transforms.RandomAffine(
            degrees=AUGMENT_CONFIG['degrees'],
            translate=AUGMENT_CONFIG['translate'],
            scale=AUGMENT_CONFIG['scale'],
            shear=AUGMENT_CONFIG['shear'],
        ),

        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def get_inference_transform(image_size=DEFAULT_IMAGE_SIZE):
    """
    The clean pipeline for VALIDATION / INFERENCE.
    No randomness, just formatting.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

# =========================================================
#      VISUALIZATION LOGIC (Runs only if executed directly)
# =========================================================
def preview_augmentations(target_folder, num_samples=5):
    if not os.path.exists(target_folder):
        print(f"Error: Folder '{target_folder}' not found.")
        return

    # Find images
    files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        print("No images found in folder.")
        return
    
    # Select random sample
    selected_files = random.sample(files, min(len(files), num_samples))
    
    # Load Transforms
    train_transform = get_train_transform()
    
    # Setup Plot
    # We use constrained_layout=False so we can manually adjust the rect later
    fig, axes = plt.subplots(len(selected_files), 6, figsize=(16, 3.5 * len(selected_files)))
    
    # Bold Title with extra padding (y=0.98 puts it high up)
    fig.suptitle(f"Augmentation Preview (Original vs 5 Random Augmentations)", fontsize=18, fontweight='bold', y=0.98)

    for i, filename in enumerate(selected_files):
        img_path = os.path.join(target_folder, filename)
        original_pil = Image.open(img_path)
        
        # --- Column 0: Original Image ---
        ax_orig = axes[i, 0] if len(selected_files) > 1 else axes[0]
        ax_orig.imshow(original_pil, cmap='gray')
        ax_orig.axis('off')
        
        # ONLY add titles to the first row to keep it clean
        if i == 0:
            ax_orig.set_title("Original", fontsize=14, fontweight='bold', pad=12)
        
        # --- Columns 1-5: Random Augmentations ---
        for j in range(1, 6):
            # Apply the transformation pipeline
            aug_tensor = train_transform(original_pil)
            
            # UN-NORMALIZE for visualization
            display_img = aug_tensor * 0.5 + 0.5
            display_img = display_img.squeeze().numpy()
            
            ax_aug = axes[i, j] if len(selected_files) > 1 else axes[j]
            ax_aug.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            ax_aug.axis('off')
            
            # Column Headers for the first row only
            if i == 0:
                ax_aug.set_title(f"Augmentation {j}", fontsize=12, pad=12)

    # CRITICAL FIX: Reserve top 5% of space for the main Suptitle
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THIS PATH TO TEST ---
    TEST_FOLDER = r"C:\Users\Aries\Desktop\Augmentation_Test\open" 
    
    print(f"Previewing augmentations from: {TEST_FOLDER}")
    preview_augmentations(TEST_FOLDER, num_samples=4)