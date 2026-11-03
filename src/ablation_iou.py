"""
Ablation B1 — Non-Maximum Suppression (NMS) IoU Threshold Comparison (0.50 vs 0.70)
Usage: python main.py --mode ablation
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS   = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
DATA_YAML = os.path.join(ROOT_DIR, "dataset", "data.yaml")
OUT_DIR   = os.path.join(ROOT_DIR, "results", "ablation")
IMGSZ     = 640
CONF      = 0.001
DEVICE    = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def run_ablation_iou():
    os.chdir(ROOT_DIR)
    model    = YOLO(WEIGHTS)
    iou_vals = [0.50, 0.70]
    rows     = []

    for iou in iou_vals:
        val = model.val(
            data    = DATA_YAML,
            imgsz   = IMGSZ,
            conf    = CONF,
            iou     = iou,
            device  = DEVICE,
            verbose = False,
        )
        p   = val.results_dict.get("metrics/precision(B)", 0)
        r   = val.results_dict.get("metrics/recall(B)",    0)
        m50 = val.results_dict.get("metrics/mAP50(B)",     0)
        m95 = val.results_dict.get("metrics/mAP50-95(B)",  0)
        f1  = 2 * p * r / (p + r + 1e-8)
        rows.append({
            "IoU": iou,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "mAP50": m50,
            "mAP50-95": m95,
        })
        print(f"IoU={iou:.2f} | P={p:.4f} R={r:.4f} F1={f1:.4f} mAP50={m50:.4f} mAP95={m95:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "iou_sweep.csv"), index=False)

    r1, r2 = rows
    metric_names = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    v1 = [r1["Precision"], r1["Recall"], r1["F1"], r1["mAP50"], r1["mAP50-95"]]
    v2 = [r2["Precision"], r2["Recall"], r2["F1"], r2["mAP50"], r2["mAP50-95"]]
    x = np.arange(len(metric_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, v1, w, label="NMS IoU=0.50", alpha=0.85, color="steelblue")
    b2 = ax.bar(x + w / 2, v2, w, label="NMS IoU=0.70", alpha=0.85, color="coral")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Ablation B1 — NMS IoU Threshold Comparison (0.50 vs 0.70)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_iou_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    lines = [
        "=== ABLATION B1 — NMS IoU COMPARISON ===",
        f"{'IoU':<8} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP95':>8}",
        "-" * 52,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['IoU']:<8.2f} {row['Precision']:>8.4f} {row['Recall']:>8.4f} "
            f"{row['F1']:>8.4f} {row['mAP50']:>8.4f} {row['mAP50-95']:>8.4f}"
        )
    with open(os.path.join(OUT_DIR, "iou_sweep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return df


if __name__ == "__main__":
    run_ablation_iou()
