"""
CE 480 Deep Learning — Project 3
Multi-Class Brain Hemorrhage Object Detection & Spatial Localization (YOLOv8)

Usage:
  python main.py --mode check
  python main.py --mode verify_labels
  python main.py --mode train
  python main.py --mode ablation
  python main.py --mode evaluate
  python main.py --mode error_analysis
  python main.py --mode gui
  python main.py --mode all
"""

import os
import sys
import argparse
import torch

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def get_device_str():
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "Apple Metal Performance Shaders (MPS)"
    return "CPU"


def run_check():
    print("\n" + "=" * 55)
    print("  YOLOv8 Brain Hemorrhage Detection — Status Check")
    print("=" * 55)

    checks = [
        ("Dataset Config (data.yaml)",      "dataset/data.yaml"),
        ("Dataset — Train Images",          "dataset/train/images"),
        ("Dataset — Validation Images",     "dataset/valid/images"),
        ("Dataset — Test Images",           "dataset/test/images"),
        ("Sample CT Scans (5 Subtypes)",    "samples"),
        ("Trained YOLOv8n Weights",         "saved_models/baseline_best.pt"),
        ("Baseline Metrics & Plots",        "results/baseline/metrics.txt"),
        ("Ablation Study Results",          "results/ablation/confidence_sweep.csv"),
        ("Test Evaluation Summary",         "results/full_test/full_test_summary.txt"),
        ("FP/FN Error Analysis Grids",      "results/error_analysis/false_positives.png"),
    ]

    for label, rel_path in checks:
        full = os.path.join(BASE_DIR, rel_path)
        if os.path.isdir(full):
            count = sum(
                len(files) for _, _, files in os.walk(full)
                if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
            )
            print(f"  [OK]  {label:<32}: {count} image(s)")
        elif os.path.exists(full):
            size_kb = os.path.getsize(full) / 1024.0
            if size_kb > 1024:
                print(f"  [OK]  {label:<32}: {size_kb / 1024.0:.2f} MB")
            else:
                print(f"  [OK]  {label:<32}: {size_kb:.1f} KB")
        else:
            print(f"  [--]  {label:<32}: not found ({rel_path})")

    print(f"\n  Hardware Accelerator : {get_device_str()}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CE 480 Project 3 — YOLOv8 Multi-Class Brain Hemorrhage Detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="check",
        choices=[
            "check",
            "verify_labels",
            "train",
            "ablation",
            "evaluate",
            "error_analysis",
            "gui",
            "all",
        ],
        help="Execution mode",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    args = parser.parse_args()

    if args.mode == "check":
        run_check()
    elif args.mode == "verify_labels":
        from src.label_verify import run_label_verify
        run_label_verify()
    elif args.mode == "train":
        from src.train import train
        train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    elif args.mode == "ablation":
        from src.ablation import run_ablation
        from src.ablation_iou import run_ablation_iou
        run_ablation_iou()
        run_ablation(train_960=False)
    elif args.mode == "evaluate":
        from src.evaluate import run_evaluate
        run_evaluate()
    elif args.mode == "error_analysis":
        from src.error_analysis import run_error_analysis
        run_error_analysis()
    elif args.mode == "gui":
        from app.app_desktop import launch_gui
        launch_gui()
    elif args.mode == "all":
        from src.label_verify import run_label_verify
        from src.train import train
        from src.ablation import run_ablation
        from src.ablation_iou import run_ablation_iou
        from src.evaluate import run_evaluate
        from src.error_analysis import run_error_analysis

        run_check()
        run_label_verify()
        train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
        run_ablation(train_960=True)
        run_ablation_iou()
        run_evaluate()
        run_error_analysis()


if __name__ == "__main__":
    main()
