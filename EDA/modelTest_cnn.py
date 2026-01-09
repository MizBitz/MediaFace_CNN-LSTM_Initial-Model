import os
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_PATH = "eye_state_mobilenet.onnx"
EVAL_DIR = r"S:\VSCode Projects\MediaFace\dataset_uncleaned"  # must contain: closed/ and open/
BATCH_SIZE = 64
IMAGE_SIZE = (64, 64)

# Threshold sweep config
THRESHOLDS = np.linspace(0.01, 0.99, 99, dtype=np.float32)

# If you want to optimize "closed precision" but keep high closed recall:
MIN_CLOSED_RECALL = 0.98  # set to None to disable constraint

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float):
    # labels: 0=closed, 1=open
    pred = (probs > thr).astype(np.int64)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)

    # Recall
    rec_open = tp / max(tp + fn, 1)
    rec_closed = tn / max(tn + fp, 1)

    # Precision
    prec_open = tp / max(tp + fp, 1)
    prec_closed = tn / max(tn + fn, 1)

    bal_acc = 0.5 * (rec_open + rec_closed)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "prec_open": float(prec_open),
        "rec_open": float(rec_open),
        "prec_closed": float(prec_closed),
        "rec_closed": float(rec_closed),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def main():
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    ds = datasets.ImageFolder(EVAL_DIR, transform=tfm)
    print("class_to_idx:", ds.class_to_idx)  # expect {'closed': 0, 'open': 1}
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Collect all probs/labels first (makes threshold sweeps easy)
    all_probs = []
    all_labels = []

    for x, y in dl:
        x_np = x.numpy().astype(np.float32)          # [B,1,64,64]
        y_np = y.numpy().astype(np.int64)            # [B]

        logits = sess.run([output_name], {input_name: x_np})[0]  # [B,1] or [B]
        logits = np.asarray(logits).reshape(-1)                  # [B]
        probs = sigmoid(logits).astype(np.float32)               # P(open)

        all_probs.append(probs)
        all_labels.append(y_np)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Baseline at 0.5 (equivalent to logit > 0)
    base = metrics_at_threshold(probs, labels, 0.5)
    print("\nBaseline (thr=0.50):")
    print(f"Acc: {base['acc']*100:.2f}% | BalAcc: {base['bal_acc']*100:.2f}%")
    print(f"Open   P:{base['prec_open']:.3f} R:{base['rec_open']:.3f}")
    print(f"Closed P:{base['prec_closed']:.3f} R:{base['rec_closed']:.3f}")
    print(f"(TP:{base['tp']} TN:{base['tn']} FP:{base['fp']} FN:{base['fn']})")

    # Sweep thresholds
    best_bal = None
    best_closed_prec = None

    for thr in THRESHOLDS:
        m = metrics_at_threshold(probs, labels, float(thr))

        if (best_bal is None) or (m["bal_acc"] > best_bal["bal_acc"]):
            best_bal = m

        if MIN_CLOSED_RECALL is None or m["rec_closed"] >= MIN_CLOSED_RECALL:
            if (best_closed_prec is None) or (m["prec_closed"] > best_closed_prec["prec_closed"]):
                best_closed_prec = m

    print("\nBest Balanced Accuracy:")
    print(f"thr={best_bal['thr']:.2f} | Acc: {best_bal['acc']*100:.2f}% | BalAcc: {best_bal['bal_acc']*100:.2f}%")
    print(f"Open   P:{best_bal['prec_open']:.3f} R:{best_bal['rec_open']:.3f}")
    print(f"Closed P:{best_bal['prec_closed']:.3f} R:{best_bal['rec_closed']:.3f}")

    if best_closed_prec is not None:
        print("\nBest Closed Precision" + (f" (rec_closed ≥ {MIN_CLOSED_RECALL})" if MIN_CLOSED_RECALL is not None else "") + ":")
        print(f"thr={best_closed_prec['thr']:.2f} | Acc: {best_closed_prec['acc']*100:.2f}% | BalAcc: {best_closed_prec['bal_acc']*100:.2f}%")
        print(f"Open   P:{best_closed_prec['prec_open']:.3f} R:{best_closed_prec['rec_open']:.3f}")
        print(f"Closed P:{best_closed_prec['prec_closed']:.3f} R:{best_closed_prec['rec_closed']:.3f}")
    else:
        print("\nNo threshold met the closed-recall constraint; lower MIN_CLOSED_RECALL or set it to None.")

if __name__ == "__main__":
    main()