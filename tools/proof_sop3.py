import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
#      1. DEFINE MODEL (Standalone Safety)
# ==========================================
# Redefining here to prevent accidental training loops from imports
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
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Latency on: {device}")
    
    model = EyeStateCNN().to(device)
    model.eval()

    # 2. Create Dummy Input (1 Frame, Grayscale, 64x64)
    dummy_input = torch.randn(1, 1, 64, 64).to(device)

    # 3. Warmup (Wake up the GPU/CPU)
    print("Warming up...")
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)

    # 4. Benchmark Loop
    latencies = []
    iterations = 1000
    print(f"Running {iterations} inference tests...")

    with torch.no_grad():
        for _ in range(iterations):
            # Start Timer
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            
            # INFERENCE
            _ = model(dummy_input)
            
            # Stop Timer
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.perf_counter()
            
            # Convert to milliseconds
            latencies.append((end - start) * 1000)

    # 5. Statistics
    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p99_lat = np.percentile(latencies, 99) # 99th percentile (worst case)

    print("\n" + "="*30)
    print("   SOP 3 RESULTS")
    print("="*30)
    print(f"Average Latency: {avg_lat:.4f} ms")
    print(f"Min Latency:     {min_lat:.4f} ms")
    print(f"Max Latency:     {max_lat:.4f} ms")
    print(f"99% Percentile:  {p99_lat:.4f} ms")

    # 6. Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
    
    # Add Reference Lines
    plt.axvline(avg_lat, color='red', linestyle='dashed', linewidth=2, label=f'Avg Speed: {avg_lat:.2f}ms')
    plt.axvline(33.33, color='green', linestyle='solid', linewidth=3, label='Real-Time Limit (33ms)')
    
    plt.title("SOP 3: System Inference Latency Distribution")
    plt.xlabel("Processing Time per Frame (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = "SOP3_Latency_Histogram.png"
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")

if __name__ == "__main__":
    main()