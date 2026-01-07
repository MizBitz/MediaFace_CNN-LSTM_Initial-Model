import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DATASET_ROOT = "dataset_dynamic_3d"
CLASSES = ["open", "closed"]

def lens_from_name(name: str) -> str:
    if name.endswith("_dyn_L.jpg"): return "L"
    if name.endswith("_dyn_R.jpg"): return "R"
    if name.endswith("_dyn_Combined.jpg"): return "Combined"
    return "unknown"

def gather_stats():
    rows = []
    for cls in CLASSES:
        folder = os.path.join(DATASET_ROOT, cls)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".jpg"):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            lens = lens_from_name(fname)
            mean_intensity = float(np.mean(img))
            std_intensity = float(np.std(img))
            lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
            h, w = img.shape
            rows.append((cls, lens, w, h, mean_intensity, std_intensity, lap_var))
    return rows

def main():
    rows = gather_stats()
    if not rows:
        print("No images found under dataset_dynamic_3d.")
        return

    classes = [r[0] for r in rows]
    lenses = [r[1] for r in rows]
    means = [r[4] for r in rows]
    lap_vars = [r[6] for r in rows]

    # Counts by class and lens
    count_cls = Counter(classes)
    count_lens = Counter(lenses)
    count_cls_lens = Counter((c, l) for c, l in zip(classes, lenses))
    unknown_count = count_lens.get("unknown", 0)

    print("Total images:", len(rows))
    print("Counts by class:", count_cls)
    print("Counts by lens:", count_lens)
    print("Unknown lens filenames:", unknown_count)

    # Plot: class x lens counts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cls_labels = CLASSES
    axes[0].bar(cls_labels, [count_cls.get(c, 0) for c in cls_labels], color=["#4caf50", "#f44336"])
    axes[0].set_title("Counts by class")
    axes[0].set_ylabel("Images")

    lens_labels = ["L", "R", "Combined", "unknown"]
    axes[1].bar(lens_labels, [count_lens.get(l, 0) for l in lens_labels], color="#2196f3")
    axes[1].set_title("Counts by lens")

    # Stacked bars: class per lens
    width = 0.2
    x = np.arange(len(lens_labels))
    for i, cls in enumerate(cls_labels):
        axes[2].bar(
            x + i*width,
            [count_cls_lens.get((cls, l), 0) for l in lens_labels],
            width=width,
            label=cls
        )
    axes[2].set_xticks(x + width / 2)
    axes[2].set_xticklabels(lens_labels)
    axes[2].set_title("Counts by lens & class")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Distributions: mean intensity and Laplacian variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(means, bins=40, color="#607d8b")
    axes[0].set_title("Mean intensity distribution")
    axes[0].set_xlabel("Mean pixel value (0-255)")

    axes[1].hist(lap_vars, bins=40, color="#9c27b0")
    axes[1].set_title("Laplacian variance (sharpness)")
    axes[1].set_xlabel("Variance")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()