# YOLOv8 Brain Hemorrhage Dataset

This directory contains the YOLOv8 configuration (`data.yaml`) and split directories (`train/`, `valid/`, `test/`) for 5-class intracranial hemorrhage bounding-box detection.

> **Note:** Curated sample CT scans for all 5 subtypes (`EDH`, `IPH`, `IVH`, `SAH`, `SDH`) are tracked directly in the repository under [`samples/`](../samples) for instant testing with the desktop GUI and unit tests.

## Expected Directory Layout

```text
dataset/
├── data.yaml          # YOLOv8 dataset configuration (5 classes)
├── train/
│   ├── images/        # 719 augmented training CT slices
│   └── labels/        # Normalized YOLO bounding-box annotations (.txt)
├── valid/
│   ├── images/        # 78 validation CT slices
│   └── labels/        # Normalized YOLO bounding-box annotations (.txt)
└── test/
    ├── images/        # 36 held-out test CT slices
    └── labels/        # Normalized YOLO bounding-box annotations (.txt)
```

## Class Definitions

| Class ID | Code | Subtype | Anatomical Location | ~BBoxes |
| :---: | :---: | :--- | :--- | :---: |
| `0` | **EDH** | Epidural Hematoma | Between skull and dura mater | ~300 |
| `1` | **IPH** | Intraparenchymal Hemorrhage | Within brain parenchyma | ~120 |
| `2` | **IVH** | Intraventricular Hemorrhage | Inside cerebral ventricles | ~80 |
| `3` | **SAH** | Subarachnoid Hemorrhage | Subarachnoid space (pia/arachnoid) | ~42 |
| `4` | **SDH** | Subdural Hematoma | Below dura mater, over cortex | ~150 |

## Dataset Source

- **Roboflow Universe:** [Brain Hemorrhage Detection Dataset (v1, YOLOv8 Format)](https://universe.roboflow.com/yolo-v5-h7oh7/brain-hemorrhage-detection-8f8o0)
