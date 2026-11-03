"""
Baseline YOLOv8n Multi-Class Hemorrhage Detection Training Pipeline
Usage: python main.py --mode train
"""

import os
import shutil
import torch
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_YAML   = os.path.join(ROOT_DIR, "dataset", "data.yaml")
MODEL       = "yolov8n.pt"
IMGSZ       = 640
EPOCHS      = 50
PATIENCE    = 10
BATCH       = 8
SEED        = 42
PROJECT     = os.path.join(ROOT_DIR, "runs")
NAME        = "baseline_640"
SAVE_PT     = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
DEVICE      = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def train(epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ):
    os.chdir(ROOT_DIR)
    print("=" * 55)
    print("STARTING BASELINE YOLOv8n TRAINING")
    print(f"  Model  : {MODEL}")
    print(f"  imgsz  : {imgsz}")
    print(f"  epochs : {epochs} (patience={PATIENCE})")
    print(f"  batch  : {batch}")
    print(f"  seed   : {SEED}")
    print(f"  device : {DEVICE}")
    print("=" * 55)

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        imgsz=imgsz,
        epochs=epochs,
        patience=PATIENCE,
        batch=batch,
        seed=SEED,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
    )

    best_ckpt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    if os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(SAVE_PT), exist_ok=True)
        shutil.copy(best_ckpt, SAVE_PT)
        print(f"Saved best weights to: {SAVE_PT}")

    _report_metrics(results, imgsz)


def _report_metrics(results, imgsz):
    try:
        map50    = results.results_dict.get("metrics/mAP50(B)",    "N/A")
        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
        prec     = results.results_dict.get("metrics/precision(B)","N/A")
        rec      = results.results_dict.get("metrics/recall(B)",   "N/A")

        if all(v != "N/A" for v in [prec, rec, map50]):
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
        else:
            f1 = "N/A"

        print("\n=== BASELINE VALIDATION METRICS ===")
        print(f"  mAP@0.5      : {round(map50,    4) if map50    != 'N/A' else 'N/A'}")
        print(f"  mAP@0.5:0.95 : {round(map50_95, 4) if map50_95 != 'N/A' else 'N/A'}")
        print(f"  Precision    : {round(prec,      4) if prec     != 'N/A' else 'N/A'}")
        print(f"  Recall       : {round(rec,       4) if rec      != 'N/A' else 'N/A'}")
        print(f"  F1-Score     : {round(f1,        4) if f1       != 'N/A' else 'N/A'}")

        out_path = os.path.join(ROOT_DIR, "results", "baseline", "metrics.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== BASELINE METRICS (Validation Set) ===\n")
            f.write(f"Model       : {MODEL}\n")
            f.write(f"imgsz       : {imgsz}\n")
            f.write(f"mAP@0.5     : {map50}\n")
            f.write(f"mAP@0.5:0.95: {map50_95}\n")
            f.write(f"Precision   : {prec}\n")
            f.write(f"Recall      : {rec}\n")
            f.write(f"F1          : {f1}\n")
        print(f"Metrics saved to: {out_path}")

    except Exception as e:
        print(f"Error reporting metrics: {e}")


if __name__ == "__main__":
    train()
