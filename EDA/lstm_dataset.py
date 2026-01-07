import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# filepath: s:\VSCode Projects\MediaFace\EDA\eda_lstm_jsonl.py
DATA_PATH = "morse_sequences.jsonl"

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def main():
    rows = load_rows(DATA_PATH)
    if not rows:
        print("No data.")
        return

    labels = [r.get("label", "?") for r in rows]
    seqs = [r.get("morse_seq", "") for r in rows]
    durations = [r.get("raw_durations", []) for r in rows]
    lengths = [len(d) for d in durations]

    print("Total samples:", len(rows))
    print("Counts by label:", Counter(labels))
    print("Sequence length (min/median/mean/max):",
          min(lengths), np.median(lengths), np.mean(lengths), max(lengths))
    all_durs = np.concatenate([np.array(d) for d in durations if d])
    print("Durations (s) min/median/mean/max:",
          all_durs.min(), np.median(all_durs), all_durs.mean(), all_durs.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Per-label counts
    lbls = list(Counter(labels).keys())
    axes[0].bar(lbls, [Counter(labels)[l] for l in lbls], color="#2196f3")
    axes[0].set_title("Counts by label")
    axes[0].tick_params(axis="x", rotation=45)

    # Sequence length distribution
    axes[1].hist(lengths, bins=range(1, max(lengths)+2), color="#4caf50", align="left")
    axes[1].set_title("Blink count per sample")
    axes[1].set_xlabel("# blinks (len(raw_durations))")

    # Duration distribution
    axes[2].hist(all_durs, bins=40, color="#9c27b0")
    axes[2].set_title("Blink durations (s)")
    axes[2].set_xlabel("Duration (s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()