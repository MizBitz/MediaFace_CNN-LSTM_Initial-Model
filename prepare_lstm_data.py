import json
import os
import random
from typing import List, Dict, Tuple

import torch
from train_lstm import BlinkSequenceDataset, train_lstm_model

DATA_PATH = "morse_sequences.jsonl"
LABEL_MAP_PATH = "lstm_label_map.json"

def load_sequences(path: str) -> List[Dict]:
    samples = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # We now require 'raw_durations' and 'label'
            if "raw_durations" in obj and "label" in obj:
                samples.append(obj)
                
    print(f"Loaded {len(samples)} samples with raw duration data.")
    return samples

def build_tensors(samples: List[Dict]) -> Tuple[List[torch.Tensor], List[int], Dict[str, int]]:
    # Create label map (A -> 0, B -> 1...)
    labels = sorted({s["label"] for s in samples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"Label map: {label_to_idx}")

    sequences: List[torch.Tensor] = []
    targets: List[int] = []

    for s in samples:
        raw_data = s["raw_durations"] # List of floats, e.g. [0.2, 0.15, 0.8]
        
        if not raw_data:
            continue

        # NORMALIZE:
        # We clamp values at 2.0 seconds to prevent outliers from skewing the model
        # and then keep them as raw seconds. The LSTM will learn that 0.2 is different from 0.8.
        normalized_data = [min(d, 2.0) for d in raw_data]

        # Shape: (Sequence_Length, 1)
        # We verify that we are passing floats, not ints
        x = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(-1)
        y = label_to_idx[s["label"]]
        
        sequences.append(x)
        targets.append(y)

    print(f"Prepared {len(sequences)} usable sequences.")
    return sequences, targets, label_to_idx

def train_from_file(path: str = DATA_PATH, val_split: float = 0.2):
    samples = load_sequences(path)
    if not samples:
        print("No samples found. Record data using mc_decoder.py first.")
        return

    sequences, targets, label_to_idx = build_tensors(samples)
    if not sequences:
        print("Failed to build tensors.")
        return

    # Shuffle and split
    indices = list(range(len(sequences)))
    random.shuffle(indices)

    split = int(len(indices) * (1.0 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [targets[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [targets[i] for i in val_idx]

    train_dataset = BlinkSequenceDataset(train_sequences, train_labels)
    val_dataset = BlinkSequenceDataset(val_sequences, val_labels)

    num_classes = len(label_to_idx)
    input_dim = 1  # We are feeding 1 feature: duration

    print(f"Training LSTM on {len(train_dataset)} train and {len(val_dataset)} val samples...")
    model = train_lstm_model(train_dataset, val_dataset, input_dim=input_dim, num_classes=num_classes)

    # Save mapping
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Saved label map to {LABEL_MAP_PATH}.")

    return model

if __name__ == "__main__":
    train_from_file()