# Trained YOLOv8 Model Weights

Unlike large classification checkpoints, the **YOLOv8n (nano)** object detection weights (`baseline_best.pt`) are compact (**6.0 MB**) and are **tracked directly in this repository** for immediate out-of-the-box inference and GUI testing.

## Included Weights

| File | Architecture | Parameters | GFLOPs | Input Size | Val mAP@0.5 | Test mAP@0.5 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `baseline_best.pt` | YOLOv8n (Anchor-Free C2f + PANet) | 3.0M | 8.1 | `640×640` | `0.5110` | `0.6618` | **Included in Repo** |

## Verification

Verify the model weights and hardware acceleration status by running:

```bash
python main.py --mode check
```
