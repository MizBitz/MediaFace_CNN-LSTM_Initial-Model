import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
import os

# ==========================================
#               CONFIGURATION
# ==========================================
# Default to dynamic 3D dataset, override with EYE_DATA_DIR if needed
DATA_DIR = os.getenv("EYE_DATA_DIR", "dataset_dynamic_3d")  
MODEL_SAVE_PATH = "eye_state_cnn.pth"
ONNX_SAVE_PATH = "eye_state_cnn.onnx"
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 25
IMAGE_SIZE = (64, 64)

# ==========================================
#           1. DATA PREPARATION
# ==========================================

# Custom Filter: We MUST ignore "Combined" images. 
# They are wide (128x64) and squashing them to 64x64 destroys features.
# We only want to train on the specific Left/Right eye crops.
def is_valid_file(path):
    filename = path.lower()
    is_image = filename.endswith(('.jpg', '.jpeg', '.png'))
    is_not_combined = "combined" not in filename
    return is_image and is_not_combined

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),     
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,)) 
])

print(f"Loading data from: {DATA_DIR}")
print("Note: Automatically filtering out 'Combined' side-by-side images.")

try:
    # UPDATED: Added is_valid_file parameter to filter data
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR, 
        transform=transform, 
        is_valid_file=is_valid_file
    )
except FileNotFoundError:
    print(f"ERROR: Could not find '{DATA_DIR}'. Run DataGen_Front.py first.")
    exit()

if len(full_dataset) == 0:
    print("ERROR: No valid images found! Check that your folder contains _L.jpg or _R.jpg files.")
    exit()

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Class-balanced sampling to handle open/closed imbalance
train_targets = [full_dataset.samples[i][1] for i in train_dataset.indices]
class_counts = Counter(train_targets)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[t] for t in train_targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes found: {full_dataset.classes}")
print(f"Training on {train_size} single-eye images, Validating on {val_size}.")

# ==========================================
#           2. MODEL ARCHITECTURE
# ==========================================

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = EyeStateCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

# ==========================================
#           3. TRAINING LOOP
# ==========================================

best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Validation Accuracy: {accuracy:.2f}%")

    # Step LR scheduler on validation accuracy
    scheduler.step(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"--> Best model saved! ({accuracy:.2f}%)")

print("Training Complete!")

# ==========================================
#      4. EXPORT TO ONNX (For Web/Mobile)
# ==========================================
print("Exporting to ONNX for Web/Mobile...")

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval() 

dummy_input = torch.randn(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

torch.onnx.export(
    model, 
    dummy_input, 
    ONNX_SAVE_PATH, 
    verbose=False,
    input_names=['input'],   
    output_names=['output'], 
    opset_version=11
)

print(f"ONNX Model saved to {ONNX_SAVE_PATH}")