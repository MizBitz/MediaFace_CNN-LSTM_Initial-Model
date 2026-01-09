import torch
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
#           CONFIGURATION
# ==========================================
# Point this to your ONNX file
MODEL_PATH = "eye_state_mobilenet.onnx" 

# Point this to your clean/validation dataset
DATA_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset" 

BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    print("Initializing SOP 1 Evaluation (ONNX Mode)...")

    # 1. Setup Data Pipeline
    # We keep the PyTorch DataLoader because it's efficient at resizing/normalizing images.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        print(f"Loaded dataset from: {DATA_DIR}")
        print(f"Classes: {dataset.classes}") # Should be ['closed', 'open']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load ONNX Model
    try:
        # Use CUDA if available, otherwise CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print(f"Successfully loaded {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Get input/output names
    input_name = session.get_inputs()[0].name
    
    # 3. Run Inference
    all_preds = []
    all_labels = []
    
    print("Running evaluation... (This may take a moment)")
    
    for inputs, labels in loader:
        # ONNX requires numpy arrays, not Tensors
        ort_inputs = {input_name: to_numpy(inputs)}
        
        # Run model
        ort_outs = session.run(None, ort_inputs)
        
        # Extract Logits (The raw score)
        logits = ort_outs[0]
        
        # Convert Logits to Class Predictions
        # Your MobileNet output is shape (Batch, 1). 
        # Logic: Logit > 0 is Open (1), Logit < 0 is Closed (0).
        if logits.shape[1] == 1:
            preds = (logits > 0).astype(int).flatten()
        else:
            # Fallback for models with 2 outputs (Softmax)
            preds = np.argmax(logits, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    # 4. Generate Report
    target_names = dataset.classes 
    
    print("\n" + "="*40)
    print("   SOP 1 ANSWER: CLASSIFICATION METRICS")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    
    # Heatmap styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title("CNN Confusion Matrix\n(SOP 1 Evidence - ONNX)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = "SOP1_Evidence_Matrix_ONNX.png"
    plt.savefig(save_path)
    print(f"\n✅ Success! Matrix saved to {save_path}")

if __name__ == "__main__":
    main()