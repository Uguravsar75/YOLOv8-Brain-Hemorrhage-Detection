import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.label_verify import yolo_to_pixel, draw_boxes
from src.error_analysis import iou, find_fp_fn_in_image, load_gt


class TestYOLOv8HemorrhagePipeline(unittest.TestCase):
    def test_yolo_to_pixel_conversion_and_clipping(self):
        x1, y1, x2, y2 = yolo_to_pixel(0.5, 0.5, 0.4, 0.2, 640, 640)
        self.assertEqual((x1, y1, x2, y2), (192, 256, 448, 384))

        # Out-of-bounds normalized coordinates should clip cleanly to image bounds
        cx1, cy1, cx2, cy2 = yolo_to_pixel(0.0, 0.0, 0.5, 0.5, 640, 640)
        self.assertEqual(cx1, 0)
        self.assertEqual(cy1, 0)
        self.assertLess(cx2, 640)
        self.assertLess(cy2, 640)

    def test_bbox_iou_calculation(self):
        box_a = [100, 100, 200, 200]
        box_b = [100, 100, 200, 200]
        box_c = [300, 300, 400, 400]
        box_d = [150, 100, 250, 200]

        self.assertAlmostEqual(iou(box_a, box_b), 1.0, places=5)
        self.assertAlmostEqual(iou(box_a, box_c), 0.0, places=5)
        # 50x100 intersection = 5000; union = 10000 + 10000 - 5000 = 15000 -> IoU = 1/3
        self.assertAlmostEqual(iou(box_a, box_d), 1.0 / 3.0, places=4)

    def test_greedy_fp_fn_matching(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            # Two GT boxes: class 0 at center (320,320, w=64, h=64) and class 1 at (100,100, w=64, h=64)
            tmp.write("0 0.5 0.5 0.1 0.1\n")
            tmp.write("1 0.15625 0.15625 0.1 0.1\n")
            lbl_path = tmp.name

        try:
            gt_loaded = load_gt(lbl_path)
            self.assertEqual(len(gt_loaded), 2)

            # One matching prediction for GT #1, and one unmatched False Positive prediction
            mock_boxes = [
                SimpleNamespace(
                    xyxy=torch.tensor([[288.0, 288.0, 352.0, 352.0]]),
                    cls=torch.tensor([0.0]),
                    conf=torch.tensor([0.85]),
                ),
                SimpleNamespace(
                    xyxy=torch.tensor([[500.0, 500.0, 580.0, 580.0]]),
                    cls=torch.tensor([4.0]),
                    conf=torch.tensor([0.45]),
                ),
            ]
            mock_preds = SimpleNamespace(boxes=mock_boxes)

            fp, fn, gt_px, pred_px = find_fp_fn_in_image(lbl_path, mock_preds, 640, 640)
            self.assertEqual(len(gt_px), 2)
            self.assertEqual(len(pred_px), 2)
            self.assertEqual(len(fp), 1)
            self.assertEqual(len(fn), 1)
        finally:
            if os.path.exists(lbl_path):
                os.remove(lbl_path)

    def test_yolov8n_model_inference_on_sample_ct(self):
        weights_path = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
        sample_path = os.path.join(ROOT_DIR, "samples", "EDH", "EDH_01.jpg")
        if not (os.path.exists(weights_path) and os.path.exists(sample_path)):
            self.skipTest("Model weights or sample image not present")

        from ultralytics import YOLO
        from PIL import Image

        model = YOLO(weights_path)
        img = np.array(Image.open(sample_path).convert("RGB"))
        res = model.predict(img, conf=0.10, iou=0.50, device="cpu", verbose=False)[0]
        self.assertIsNotNone(res.boxes)
        self.assertGreaterEqual(len(res.boxes), 1)


if __name__ == "__main__":
    unittest.main()
