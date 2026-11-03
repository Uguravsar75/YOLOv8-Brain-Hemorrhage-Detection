"""
Controlled Single-Variable Ablation Studies:
  - Ablation A : Input Resolution (imgsz 640 → 960)
  - Ablation B2: Confidence Threshold Sweep (0.001, 0.01, 0.05, 0.10, 0.20)
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_YAML        = os.path.join(ROOT_DIR, "dataset", "data.yaml")
BASELINE_WEIGHTS = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
OUT_DIR          = os.path.join(ROOT_DIR, "results", "ablation")
SEED             = 42
EPOCHS           = 50
PATIENCE         = 10
DEVICE           = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def ablation_imgsz(baseline: dict):
    """Ablation A: Train YOLOv8n at imgsz=960 and plot comparison against baseline (640)."""
    os.chdir(ROOT_DIR)
    print("\n" + "=" * 50)
    print(f"ABLATION A — High Resolution imgsz=960 (device={DEVICE})")
    print("=" * 50)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data     = DATA_YAML,
        imgsz    = 960,
        epochs   = EPOCHS,
        patience = PATIENCE,
        batch    = 8,
        seed     = SEED,
        device   = DEVICE,
        project  = os.path.join(ROOT_DIR, "runs"),
        name     = "ablation_960",
        exist_ok = True,
        verbose  = True,
    )

    rd  = results.results_dict
    p   = rd.get("metrics/precision(B)", 0)
    r   = rd.get("metrics/recall(B)",    0)
    m50 = rd.get("metrics/mAP50(B)",     0)
    m95 = rd.get("metrics/mAP50-95(B)",  0)
    f1  = 2 * p * r / (p + r + 1e-8)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ablation_960_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=== ABLATION A METRICS (imgsz=960, Validation Set) ===\n")
        f.write(f"Model: YOLOv8n | imgsz: 960 | patience={PATIENCE} | seed={SEED}\n")
        f.write(f"mAP@0.5     : {m50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {m95:.4f}\n")
        f.write(f"Precision   : {p:.4f}\n")
        f.write(f"Recall      : {r:.4f}\n")
        f.write(f"F1          : {f1:.4f}\n")

    labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(
        x - w / 2,
        [baseline["mAP50"], baseline["mAP95"], baseline["P"], baseline["R"], baseline["F1"]],
        w, label="Baseline (640)", alpha=0.85, color="steelblue"
    )
    b2 = ax.bar(
        x + w / 2,
        [m50, m95, p, r, f1],
        w, label="Ablation A (960)", alpha=0.85, color="darkorange"
    )
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Ablation A — Input Resolution Comparison (640 vs 960)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_imgsz_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nmAP@0.5={m50:.4f} | mAP@0.5:0.95={m95:.4f} | P={p:.4f} | R={r:.4f} | F1={f1:.4f}")
    return {"imgsz": 960, "mAP50": m50, "mAP95": m95, "P": p, "R": r, "F1": f1}


def ablation_confidence():
    """Ablation B2: Confidence threshold sweep on validation set."""
    os.chdir(ROOT_DIR)
    print("\n" + "=" * 50)
    print(f"ABLATION B2 — Confidence Threshold Sweep (device={DEVICE})")
    print("=" * 50)

    model       = YOLO(BASELINE_WEIGHTS)
    conf_values = [0.001, 0.01, 0.05, 0.10, 0.20]
    rows        = []

    for conf in conf_values:
        val = model.val(
            data    = DATA_YAML,
            conf    = conf,
            iou     = 0.5,
            imgsz   = 640,
            device  = DEVICE,
            verbose = False,
        )
        p   = val.results_dict.get("metrics/precision(B)", 0)
        r   = val.results_dict.get("metrics/recall(B)",    0)
        m50 = val.results_dict.get("metrics/mAP50(B)",     0)
        m95 = val.results_dict.get("metrics/mAP50-95(B)",  0)
        f1  = 2 * p * r / (p + r + 1e-8)
        rows.append({
            "conf": conf,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "mAP50": m50,
            "mAP50-95": m95,
        })
        print(f"conf={conf:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f} mAP50={m50:.3f}")

    df   = pd.DataFrame(rows)
    best = df.loc[df["F1"].idxmax()]
    print(f"\nBest F1 → conf={best['conf']} : F1={best['F1']:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "confidence_sweep.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df["conf"], df["Precision"], "o-", color="blue",  label="Precision")
    ax1.plot(df["conf"], df["Recall"],    "s-", color="green", label="Recall")
    ax1.plot(df["conf"], df["F1"],        "^-", color="red",   label="F1")
    ax1.axvline(
        best["conf"], color="red", linestyle="--", alpha=0.6,
        label=f"Best F1 conf={best['conf']}"
    )
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 vs Confidence")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df["conf"], df["mAP50"],    "D-", color="purple", label="mAP@0.5")
    ax2.plot(df["conf"], df["mAP50-95"], "v-", color="orange", label="mAP@0.5:0.95")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("mAP")
    ax2.set_title("mAP vs Confidence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Ablation B2 — Confidence Threshold Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_confidence_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUT_DIR}/confidence_sweep.csv & ablation_confidence_sweep.png")
    return df


def run_ablation(train_960=False):
    baseline = {
        "mAP50": 0.4721, "mAP95": 0.2414,
        "P": 0.8688, "R": 0.3797, "F1": 0.5284
    }
    if train_960:
        ablation_imgsz(baseline)
    return ablation_confidence()


if __name__ == "__main__":
    run_ablation(train_960=True)
