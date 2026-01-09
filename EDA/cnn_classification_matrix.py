import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = "eye_state_mobilenet.onnx"
DATA_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset"  # Ensure this points to your dataset folder
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)

# ==========================================
#      1. DEFINE MODEL CLASS (STANDALONE)
# ==========================================
# We define this here to avoid importing 'train_cnn.py', 
# which would accidentally trigger a re-training loop.
class EyeStateCNN(nn.Module):
    def __init__(self):
        super(EyeStateCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Fully Connected Block
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, 2) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
#           MAIN EXECUTION
# ==========================================
def main():
    print("Initializing SOP 1 Evaluation...")

    # 1. Setup Data
    # We use the same transforms as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure '{DATA_DIR}' exists.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeStateCNN().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Successfully loaded {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Make sure it's in the same folder.")
        return

    model.eval()
    
    # 3. Predict
    all_preds = []
    all_labels = []
    
    print("Running evaluation on dataset...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Generate Report
    target_names = dataset.classes # Should be ['closed', 'open']
    print("\n" + "="*40)
    print("   SOP 1 ANSWER: CLASSIFICATION METRICS")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("CNN Confusion Matrix (SOP 1 Evidence)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = "SOP1_Evidence_Matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Success! Matrix saved to {save_path}")
    print("You can now insert this image and the numbers above into Chapter 4.")

if __name__ == "__main__":
    main()