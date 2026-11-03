"""
Spatial Bounding-Box Error Analysis (Greedy IoU Matching for FP / FN Identification)
Usage: python main.py --mode error_analysis
"""

import os
import random
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS   = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
OUT_DIR   = os.path.join(ROOT_DIR, "results", "error_analysis")
CONF_LIST = [0.10, 0.05, 0.01]
NMS_IOU   = 0.5
IOU_MATCH = 0.5
N_SAMPLES = 6
SEED      = 42
DEVICE    = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

IMG_DIRS = [
    (os.path.join(ROOT_DIR, "dataset", "valid", "images"), os.path.join(ROOT_DIR, "dataset", "valid", "labels")),
    (os.path.join(ROOT_DIR, "dataset", "test", "images"),  os.path.join(ROOT_DIR, "dataset", "test", "labels")),
]

CLASS_NAMES = ["EDH", "IPH", "IVH", "SAH", "SDH"]
COLOR_GT    = (0, 200, 0)     # Green — Ground Truth
COLOR_PRED  = (0, 0,   220)   # Red   — Model Prediction


def yolo_to_pixel(cx, cy, w, h, iw, ih):
    x1 = max(0, int((cx - w / 2) * iw))
    y1 = max(0, int((cy - h / 2) * ih))
    x2 = min(iw - 1, int((cx + w / 2) * iw))
    y2 = min(ih - 1, int((cy + h / 2) * ih))
    return x1, y1, x2, y2


def load_gt(lbl_path):
    """Parse YOLO label file and return list of [cls, cx, cy, w, h] ground-truth boxes."""
    boxes = []
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        with open(lbl_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    boxes.append([int(p[0])] + [float(x) for x in p[1:]])
    return boxes


def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter + 1e-8)


def find_fp_fn_in_image(lbl_path, preds, iw, ih):
    """
    Identify bounding-box level False Positives (FP) and False Negatives (FN)
    using greedy 1-to-1 IoU matching (threshold = IOU_MATCH).
    """
    gt_boxes   = load_gt(lbl_path)
    pred_boxes = []
    if preds.boxes is not None and len(preds.boxes) > 0:
        for box in preds.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pred_boxes.append((int(box.cls[0]), float(box.conf[0]), x1, y1, x2, y2))

    gt_pixels = []
    for g in gt_boxes:
        c, cx, cy, w, h = g
        x1, y1, x2, y2  = yolo_to_pixel(cx, cy, w, h, iw, ih)
        gt_pixels.append((c, x1, y1, x2, y2))

    matched_gt   = set()
    matched_pred = set()

    for pi, (pc, pconf, px1, py1, px2, py2) in enumerate(pred_boxes):
        best_iou = 0.0
        best_gi  = -1
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_pixels):
            if gi in matched_gt:
                continue
            v = iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
            if v > best_iou:
                best_iou = v
                best_gi  = gi
        if best_iou >= IOU_MATCH and best_gi != -1:
            matched_pred.add(pi)
            matched_gt.add(best_gi)

    fp_preds = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    fn_gts   = [gt_pixels[i]  for i in range(len(gt_pixels))  if i not in matched_gt]

    return fp_preds, fn_gts, gt_pixels, pred_boxes


def draw_fp(img_bgr, fp_preds, all_gt):
    img = img_bgr.copy()
    for (c, gx1, gy1, gx2, gy2) in all_gt:
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), COLOR_GT, 2)
        cv2.putText(img, f"GT:{CLASS_NAMES[c]}", (gx1, max(gy1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1)
    for (c, conf, px1, py1, px2, py2) in fp_preds:
        cv2.rectangle(img, (px1, py1), (px2, py2), COLOR_PRED, 2)
        cv2.putText(img, f"FP:{CLASS_NAMES[c]}({conf:.2f})", (px1, py2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_PRED, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_fn(img_bgr, fn_gts, all_preds):
    img = img_bgr.copy()
    for (c, gx1, gy1, gx2, gy2) in fn_gts:
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), COLOR_GT, 2)
        cv2.putText(img, f"FN:{CLASS_NAMES[c]}", (gx1, max(gy1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1)
    for (c, conf, px1, py1, px2, py2) in all_preds:
        cv2.rectangle(img, (px1, py1), (px2, py2), COLOR_PRED, 2)
        cv2.putText(img, f"P:{CLASS_NAMES[c]}({conf:.2f})", (px1, py2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_PRED, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_examples(model):
    all_images = []
    for img_dir, lbl_dir in IMG_DIRS:
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((img_dir, lbl_dir, fname))

    fp_examples = []
    fn_examples = []

    for conf in CONF_LIST:
        imgs = list(all_images)
        random.seed(SEED)
        random.shuffle(imgs)
        fp_examples = []
        fn_examples = []

        for img_dir, lbl_dir, fname in imgs:
            if len(fp_examples) >= N_SAMPLES and len(fn_examples) >= N_SAMPLES:
                break

            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            ih, iw = img_bgr.shape[:2]

            preds = model.predict(
                img_path, conf=conf, iou=NMS_IOU, device=DEVICE, verbose=False
            )[0]
            fp_preds, fn_gts, all_gt, all_preds = find_fp_fn_in_image(
                lbl_path, preds, iw, ih
            )

            if fp_preds and len(fp_examples) < N_SAMPLES:
                vis = draw_fp(img_bgr, fp_preds, all_gt)
                fp_examples.append((
                    fname, vis,
                    f"FP: {len(fp_preds)} bbox  |  GT: {len(all_gt)} bbox"
                ))

            if fn_gts and len(fn_examples) < N_SAMPLES:
                vis = draw_fn(img_bgr, fn_gts, all_preds)
                fn_examples.append((
                    fname, vis,
                    f"FN: {len(fn_gts)} bbox  |  Pred: {len(all_preds)}"
                ))

        print(f"conf={conf:.2f} → FP: {len(fp_examples)}, FN: {len(fn_examples)}")
        if len(fp_examples) >= N_SAMPLES and len(fn_examples) >= N_SAMPLES:
            break

    return fp_examples, fn_examples


def save_grid(examples, title, out_path):
    n    = len(examples)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    fig.patch.set_facecolor("black")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="white")

    ax_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (fname, vis, info) in enumerate(examples):
        ax = ax_flat[i]
        ax.imshow(vis)
        ax.set_title(f"Example {i+1}\n{info}", fontsize=8, color="white")
        ax.axis("off")
        ax.set_facecolor("black")

    for j in range(n, len(ax_flat)):
        ax_flat[j].axis("off")
        ax_flat[j].set_facecolor("black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved: {out_path}")


def run_error_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS)

    print("Collecting FP/FN examples across validation and test sets...")
    fp_examples, fn_examples = collect_examples(model)
    print(f"Found FP examples: {len(fp_examples)}, FN examples: {len(fn_examples)}")

    if fp_examples:
        save_grid(
            fp_examples,
            "False Positive Examples  (Green=Ground Truth  |  Red=False Prediction)",
            os.path.join(OUT_DIR, "false_positives.png")
        )
    if fn_examples:
        save_grid(
            fn_examples,
            "False Negative Examples  (Green=Missed Ground Truth  |  Red=Model Prediction)",
            os.path.join(OUT_DIR, "false_negatives.png")
        )


if __name__ == "__main__":
    run_error_analysis()
