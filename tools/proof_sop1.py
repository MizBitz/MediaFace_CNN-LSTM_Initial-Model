import torch
import onnxruntime as ort
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import data_augmentation as da

# ================= CONFIGURATION =================
# Point this to your existing ONNX file
MODEL_PATH = "eye_state_mobilenet.onnx" 

# Use the 'unseen' or 'test' dataset if you have it. 
# Otherwise, use your main dataset folder.
DATA_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset" 
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
# =================================================

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def evaluate_onnx(session, loader, title, save_matrix_name):
    print(f"\nRunning {title}...")
    
    all_preds = []
    all_labels = []
    
    # Get the name of the input node for the ONNX model
    input_name = session.get_inputs()[0].name
    
    for inputs, labels in loader:
        # 1. Convert PyTorch tensor to Numpy (ONNX format)
        ort_inputs = {input_name: to_numpy(inputs)}
        
        # 2. Run Inference
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0] # The model output
        
        # 3. Convert Logits to Class Predictions
        # (Logits > 0 means Open, < 0 means Closed)
        # Note: Depending on your export, it might output shape (N, 1) or (N, 2).
        # Your train script outputted (N, 1) where >0 is Open.
        
        if logits.shape[1] == 1:
            # Binary Logit Case
            preds = (logits > 0).astype(int).flatten()
        else:
            # Softmax/Argmax Case (just in case)
            preds = np.argmax(logits, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    # Report
    target_names = ['closed', 'open']
    print(f"\n--- {title} Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"{title}\n(Confusion Matrix)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_matrix_name)
    print(f"✅ Matrix saved to {save_matrix_name}")

def main():
    print("Initializing SOP 1 & 4 Evaluation (ONNX Mode)...")

    # 1. Load ONNX Model
    try:
        # Providers: try CUDA if available, else CPU
        ort_session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"Successfully loaded {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # 2. Setup Data Loaders
    # A. Clean Data (SOP 1 - Baseline)
    transform_clean = da.get_inference_transform(IMAGE_SIZE)
    ds_clean = datasets.ImageFolder(root=DATA_DIR, transform=transform_clean)
    loader_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)
    
    # B. Stressed Data (SOP 4 - Robustness)
    transform_stress = da.get_train_transform(IMAGE_SIZE)
    ds_stress = datasets.ImageFolder(root=DATA_DIR, transform=transform_stress)
    loader_stress = DataLoader(ds_stress, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Run Evaluations
    evaluate_onnx(
        ort_session, 
        loader_clean, 
        title="SOP 1: Baseline Accuracy (Ideal)", 
        save_matrix_name="SOP1_Baseline_Matrix.png"
    )

    evaluate_onnx(
        ort_session, 
        loader_stress, 
        title="SOP 4: Robustness Test (Noise/Blur)", 
        save_matrix_name="SOP4_Robustness_Matrix.png"
    )

    print("\nDone! Images saved.")

if __name__ == "__main__":
    main()