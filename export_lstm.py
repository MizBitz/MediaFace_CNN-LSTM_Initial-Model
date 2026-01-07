import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
import json
from train_lstm import BlinkLSTM

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = "blink_lstm.pth"
ONNX_PATH = "blink_lstm.onnx"
LABEL_MAP_PATH = "lstm_label_map.json"

# ==========================================
#      1. DEFINE THE WRAPPER
# ==========================================
class ExportableBlinkLSTM(torch.nn.Module):
    """
    A lightweight wrapper that removes the 'pack_padded_sequence' logic
    so the model can be exported to ONNX for Web/Mobile use.
    """
    def __init__(self, base_model):
        super().__init__()
        # We assume the base_model is already trained.
        # We share the specific layers (pointers) so no weights are copied/duplicated.
        self.lstm = base_model.lstm
        self.fc = base_model.fc
        
    def forward(self, x, lengths):
        # x shape: (Batch, Seq_Len, Features)
        
        # 1. Run LSTM directly (No packing/padding logic needed for Batch=1)
        # We ignore the 'lengths' argument during inference because we process
        # the entire input sequence as valid data.
        _, (h_n, _) = self.lstm(x)
        
        # 2. Reconstruct the feature vector from the hidden states
        # The trained model used: h_final = cat(h_n[-2], h_n[-1])
        # We must replicate this EXACTLY.
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        # 3. Classify
        logits = self.fc(h_final)
        return logits

# ==========================================
#           MAIN EXECUTION
# ==========================================
def main():
    # 1. Load Label Map
    print("Loading label map...")
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
        num_classes = len(label_map)

    # 2. Load Base Model
    print("Loading trained PyTorch model...")
    device = torch.device("cpu")
    base_model = BlinkLSTM(input_dim=1, hidden_dim=64, num_layers=2, num_classes=num_classes)
    
    # Load weights (suppress security warning for local file)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    
    # 3. Wrap Model
    print("Wrapping model for export...")
    export_model = ExportableBlinkLSTM(base_model)
    export_model.eval()

    # 4. Create Dummy Input
    # Batch=1, Sequence Length=5, Features=1
    dummy_input = torch.randn(1, 5, 1)
    dummy_lengths = torch.tensor([5], dtype=torch.long)

    # 5. Export to ONNX
    print(f"Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        export_model, 
        (dummy_input, dummy_lengths), 
        ONNX_PATH, 
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input', 'lengths'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✅ Export successful.")

    # 6. CONSISTENCY CHECK (PyTorch vs ONNX)
    print("\nRunning consistency check...")
    
    # A. Get PyTorch Output
    with torch.no_grad():
        pt_logits = export_model(dummy_input, dummy_lengths).numpy()

    # B. Get ONNX Output
    ort_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_inputs = {i.name: i for i in ort_sess.get_inputs()}

    feed = {"input": dummy_input.numpy().astype(np.float32)}
    # Some exports drop the unused 'lengths' input during optimization.
    if "lengths" in ort_inputs:
        feed["lengths"] = dummy_lengths.numpy().astype(np.int64)

    ort_logits = ort_sess.run(None, feed)[0]

    # C. Compare
    print("PyTorch sample: ", pt_logits[0][:3])
    print("ONNX sample:    ", ort_logits[0][:3])
    
    diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"\nMax Absolute Difference: {diff:.8f}")

    if diff < 1e-4:
        print("✅ PASSED: Models are identical.")
    else:
        print("⚠️ WARNING: Models diverge. Check opset version or LSTM logic.")

if __name__ == "__main__":
    main()