"""
Ground-Truth YOLO Bounding-Box Verification & 4x5 Gallery Generator
Usage: python main.py --mode verify_labels
"""

import os
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "train")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "results", "label_gallery")
NUM_IMAGES  = 20
SEED        = 42

CLASS_NAMES = ["EDH", "IPH", "IVH", "SAH", "SDH"]

# Distinct colors per hemorrhage subtype (BGR format for OpenCV)
COLORS = {
    0: (0,   0,   255),   # EDH → Red
    1: (0,   255, 0  ),   # IPH → Green
    2: (0,   165, 255),   # IVH → Orange
    3: (255, 0,   255),   # SAH → Magenta
    4: (0,   255, 255),   # SDH → Yellow
}


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO (cx, cy, w, h) coordinates to pixel (x1, y1, x2, y2)."""
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w - 1, int((cx + w / 2) * img_w))
    y2 = min(img_h - 1, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def draw_boxes(image_path, label_path):
    """Render ground-truth bounding boxes and class labels onto a CT image."""
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    img_h, img_w = img.shape[:2]
    classes_found = []

    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
                color = COLORS.get(cls, (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label_text = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                y_top = max(y1, th + 6)
                cv2.rectangle(img, (x1, y_top - th - 6), (x1 + tw + 4, y_top), color, -1)
                cv2.putText(
                    img, label_text, (x1 + 2, y_top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )
                classes_found.append(cls)
    else:
        cv2.putText(
            img, "NEGATIVE SAMPLE", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), classes_found


def save_gallery(selected, img_dir, lbl_dir, output_dir=OUTPUT_DIR):
    """Save 20 randomly sampled training images as a 4x5 verification grid."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(
        "Label Verification Gallery — Multi-Class Brain Hemorrhage Detection (YOLOv8)",
        fontsize=14, fontweight="bold"
    )

    stats = {name: 0 for name in CLASS_NAMES}
    negative_count = 0

    for idx, fname in enumerate(selected):
        ax = axes[idx // 5][idx % 5]
        img_path = os.path.join(img_dir, fname)
        base_name = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        img_drawn, classes = draw_boxes(img_path, lbl_path)

        if img_drawn is None:
            ax.axis("off")
            continue

        ax.imshow(img_drawn)
        ax.axis("off")

        if classes:
            title = ", ".join(sorted(set(CLASS_NAMES[c] for c in classes)))
            ax.set_title(title, fontsize=8, color="red")
            for c in classes:
                stats[CLASS_NAMES[c]] += 1
        else:
            ax.set_title("Negative", fontsize=8, color="gray")
            negative_count += 1

    patches = [
        mpatches.Patch(color=np.array(COLORS[i][::-1]) / 255, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    patches.append(mpatches.Patch(color="gray", label="Negative"))
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = os.path.join(output_dir, "label_gallery.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gallery saved to: {out_path}")

    return stats, negative_count


def run_label_verify():
    img_dir = os.path.join(DATASET_DIR, "images")
    lbl_dir = os.path.join(DATASET_DIR, "labels")
    if not os.path.isdir(img_dir):
        print(f"Dataset directory not found: {img_dir}")
        return

    all_images = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    random.seed(SEED)
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

    print(f"Processing {len(selected)} sampled training images...")
    stats, neg = save_gallery(selected, img_dir, lbl_dir)

    print("\n=== CLASS DISTRIBUTION IN 20 SAMPLED IMAGES ===")
    for name, count in stats.items():
        print(f"  {name}: {count} bbox")
    print(f"  Negative images: {neg}")


if __name__ == "__main__":
    run_label_verify()
