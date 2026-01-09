import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
#           CONFIGURATION
# ==========================================
DATA_DIR = r"S:\VSCode Projects\MediaFace\dataset_dynamic_aligned"
BATCH_SIZE = 32
EPOCHS = 20  # Enough to show convergence
LR = 0.001

# ==========================================
#        MODEL DEFINITION (Copy)
# ==========================================
class EyeStateCNN(nn.Module):
    def __init__(self):
        super(EyeStateCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
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

def main():
    print("Preparing Data...")
    
    # 1. Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. Load Dataset
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 3. Handle Imbalance (Weighted Sampler)
    # We need to calculate weights for the subset, which is tricky, 
    # so for this graph visualization, a standard loader is fine 
    # as long as we see the loss going down.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeStateCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- STORAGE FOR GRAPH ---
    train_losses = []
    val_losses = []
    
    print(f"Starting Training for {EPOCHS} epochs to generate Loss Curve...")
    
    for epoch in range(EPOCHS):
        # TRAIN
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
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # VALIDATE
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # ==========================================
    #           PLOTTING
    # ==========================================
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    
    # Plot Lines
    plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    
    # Labels
    plt.title("CNN Training vs. Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (CrossEntropy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = "training_loss_curve.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()