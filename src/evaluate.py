"""
Validation & Held-Out Test Set Evaluation (Per-Class & Overall mAP Breakdown)
Usage: python main.py --mode evaluate
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS     = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
DATA_YAML   = os.path.join(ROOT_DIR, "dataset", "data.yaml")
CLASS_NAMES = ["EDH", "IPH", "IVH", "SAH", "SDH"]
OUT         = os.path.join(ROOT_DIR, "results", "full_test")
DEVICE      = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def per_class(metrics):
    n = len(CLASS_NAMES)
    def safe(arr):
        try:
            lst = [float(v) for v in list(arr)]
            return (lst + [0.0] * n)[:n]
        except Exception:
            return [0.0] * n
    return safe(metrics.box.maps), safe(metrics.box.p), safe(metrics.box.r)


def overall(m):
    p   = m.results_dict.get("metrics/precision(B)", 0)
    r   = m.results_dict.get("metrics/recall(B)",    0)
    m50 = m.results_dict.get("metrics/mAP50(B)",     0)
    m95 = m.results_dict.get("metrics/mAP50-95(B)",  0)
    f1  = 2 * p * r / (p + r + 1e-8)
    return p, r, f1, m50, m95


def run_evaluate():
    os.chdir(ROOT_DIR)
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(WEIGHTS)

    EVAL_CONF = 0.001

    print("\n" + "=" * 55)
    print(f"VALIDATION SET EVALUATION (device={DEVICE})")
    print("=" * 55)
    val_m = model.val(
        data=DATA_YAML, split="val", imgsz=640,
        conf=EVAL_CONF, iou=0.5, device=DEVICE, verbose=True
    )

    print("\n" + "=" * 55)
    print(f"HELD-OUT TEST SET EVALUATION (device={DEVICE})")
    print("=" * 55)
    tst_m = model.val(
        data=DATA_YAML, split="test", imgsz=640,
        conf=EVAL_CONF, iou=0.5, device=DEVICE, verbose=True
    )

    val_maps, val_ps, val_rs = per_class(val_m)
    tst_maps, tst_ps, tst_rs = per_class(tst_m)
    vp, vr, vf, vm50, vm95   = overall(val_m)
    tp, tr, tf, tm50, tm95   = overall(tst_m)

    print("\n" + "=" * 60)
    print("OVERALL METRICS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Val (640)':>12} {'Test (640)':>12}")
    print("-" * 46)
    for label, v, t in [
        ("mAP@0.5", vm50, tm50),
        ("mAP@0.5:0.95", vm95, tm95),
        ("Precision", vp, tp),
        ("Recall", vr, tr),
        ("F1", vf, tf),
    ]:
        print(f"{label:<20} {v:>12.4f} {t:>12.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS mAP@0.5")
    print("=" * 60)
    print(f"{'Class':<8} {'Val':>8} {'Test':>8}")
    print("-" * 26)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:<8} {val_maps[i]:>8.4f} {tst_maps[i]:>8.4f}")

    # Per-class bar chart
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Per-Class Performance — Validation vs Test", fontsize=13, fontweight="bold")

    for ax, vals_v, vals_t, title in [
        (axes[0], val_maps, tst_maps, "mAP@0.5"),
        (axes[1], val_ps,   tst_ps,   "Precision"),
        (axes[2], val_rs,   tst_rs,   "Recall"),
    ]:
        b1 = ax.bar(x - w / 2, vals_v, w, label="Val",  color="steelblue",  alpha=0.85)
        b2 = ax.bar(x + w / 2, vals_t, w, label="Test", color="darkorange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bar in [*b1, *b2]:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5
                )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_class_val_vs_test.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Overall summary bar chart
    metrics_lbl = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    val_vals    = [vm50, vm95, vp, vr, vf]
    test_vals   = [tm50, tm95, tp, tr, tf]

    x2 = np.arange(len(metrics_lbl))
    w2 = 0.35
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    b1 = ax2.bar(x2 - w2 / 2, val_vals,  w2, label="Validation", color="steelblue",  alpha=0.9)
    b2 = ax2.bar(x2 + w2 / 2, test_vals, w2, label="Test",       color="darkorange", alpha=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_lbl, fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Score")
    ax2.set_title("Baseline YOLOv8n — Overall Performance (Val vs Test)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bar in [*b1, *b2]:
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "overall_val_vs_test.png"), dpi=150, bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUT, "full_test_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== FULL EVALUATION SUMMARY (VAL vs TEST) ===\n\n")
        f.write(f"{'Metric':<20} {'Val':>10} {'Test':>10}\n")
        f.write("-" * 42 + "\n")
        for label, v, t in [
            ("mAP@0.5", vm50, tm50),
            ("mAP@0.5:0.95", vm95, tm95),
            ("Precision", vp, tp),
            ("Recall", vr, tr),
            ("F1", vf, tf),
        ]:
            f.write(f"{label:<20} {v:>10.4f} {t:>10.4f}\n")
        f.write("\nPer-Class mAP@0.5:\n")
        f.write(f"{'Class':<8} {'Val':>8} {'Test':>8}\n")
        f.write("-" * 26 + "\n")
        for i, cls in enumerate(CLASS_NAMES):
            f.write(f"{cls:<8} {val_maps[i]:>8.4f} {tst_maps[i]:>8.4f}\n")

    print(f"\nCharts and summary saved to: {OUT}/")


if __name__ == "__main__":
    run_evaluate()
