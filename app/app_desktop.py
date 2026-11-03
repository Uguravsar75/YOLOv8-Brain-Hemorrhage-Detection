# -*- coding: utf-8 -*-
"""
PyQt Dark-Mode Clinical Decision-Support Desktop GUI for Multi-Class Brain Hemorrhage Detection
Usage: python main.py --mode gui
"""

import sys
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSlider, QFileDialog, QTextEdit, QSizePolicy, QFrame,
    )
    from PyQt5.QtGui import QImage, QPixmap, QFont
    from PyQt5.QtCore import Qt
    QT_VER = 5
except ImportError:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSlider, QFileDialog, QTextEdit, QSizePolicy, QFrame,
    )
    from PyQt6.QtGui import QImage, QPixmap, QFont
    from PyQt6.QtCore import Qt
    QT_VER = 6

# ── constants ────────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS     = os.path.join(ROOT_DIR, "saved_models", "baseline_best.pt")
DEVICE      = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = ["EDH", "IPH", "IVH", "SAH", "SDH"]
CLASS_FULL  = {
    "EDH": "Epidural Hematoma",
    "IPH": "Intraparenchymal Hemorrhage",
    "IVH": "Intraventricular Hemorrhage",
    "SAH": "Subarachnoid Hemorrhage",
    "SDH": "Subdural Hematoma",
}
CLASS_LOC = {
    "EDH": "Between skull and dura mater",
    "IPH": "Within brain parenchyma",
    "IVH": "Inside cerebral ventricles",
    "SAH": "Subarachnoid space (pia/arachnoid)",
    "SDH": "Below dura mater, over cortex",
}
COLORS_RGB = {
    "EDH": (220,  50,  50),   # Red
    "IPH": ( 50, 180,  50),   # Green
    "IVH": (255, 140,   0),   # Orange
    "SAH": (180,   0, 180),   # Magenta
    "SDH": (  0, 180, 200),   # Cyan
}

BG        = "#0d1117"
BG2       = "#161b22"
BORDER    = "#21262d"
FG        = "#c9d1d9"
FG_BRIGHT = "#e6edf3"
FG_DIM    = "#8b949e"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG}; }}
QPushButton {{
    background: #238636; color: #fff;
    border: 1px solid #2ea043; border-radius: 6px;
    padding: 8px 18px; font-size: 13px; font-weight: bold;
}}
QPushButton:hover {{ background: #2ea043; }}
QSlider::groove:horizontal {{ height:4px; background:{BORDER}; border-radius:2px; }}
QSlider::handle:horizontal {{ width:14px; height:14px; margin:-5px 0; background:{ACCENT}; border-radius:7px; }}
QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
QTextEdit {{
    background: {BG2}; color: {FG};
    border: 1px solid {BORDER}; border-radius: 6px; font-size: 13px;
}}
"""

# ── model ────────────────────────────────────────────────────────────────────

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(WEIGHTS)
    return _MODEL


def run_detect(image_rgb, conf=0.10, iou=0.50):
    model = get_model()
    res = model.predict(image_rgb, conf=conf, iou=iou, device=DEVICE, verbose=False)[0]
    img = image_rgb.copy()
    dets = []
    if res.boxes is not None:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            score = float(box.conf[0])
            name = CLASS_NAMES[cls]
            color = COLORS_RGB[name]
            wb, hb = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            lbl_txt = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y_top = max(y1, th + 8)
            cv2.rectangle(img, (x1, y_top - th - 8), (x1 + tw + 6, y_top), color, -1)
            cv2.putText(
                img, lbl_txt, (x1 + 3, y_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            dets.append((name, score, x1, y1, x2, y2, wb, hb))
    return img, dets


# ── helpers ──────────────────────────────────────────────────────────────────

def numpy_to_pixmap(arr_rgb):
    arr_rgb = np.ascontiguousarray(arr_rgb)
    h, w, ch = arr_rgb.shape
    fmt = QImage.Format_RGB888 if QT_VER == 5 else QImage.Format.Format_RGB888
    return QPixmap.fromImage(QImage(arr_rgb.data, w, h, ch * w, fmt))


def mkl(text, bold=False, color=None, size=12):
    l = QLabel(text)
    f = QFont()
    f.setPointSize(size)
    if bold:
        f.setBold(True)
    l.setFont(f)
    if color:
        l.setStyleSheet(f"color:{color}; background:transparent;")
    return l


SIMPLE_DESC = {
    "EDH": "Epidural Hematoma — Blood accumulation between the skull and dura mater. Usually caused by head trauma; requires emergency intervention.",
    "IPH": "Intraparenchymal Hemorrhage — Bleeding within the brain tissue itself. May result from stroke, hypertension, or trauma.",
    "IVH": "Intraventricular Hemorrhage — Bleeding into the fluid-filled ventricles of the brain. Common in premature infants and hypertensive patients.",
    "SAH": "Subarachnoid Hemorrhage — Bleeding between the membranes covering the brain surface. Typically presents with sudden severe headache.",
    "SDH": "Subdural Hematoma — Blood pooling below the dura mater, over the brain surface. Can occur in elderly patients even after minor head injuries.",
}


def build_technical(dets, conf_thr, iou_thr, img_shape):
    if not dets:
        return (
            "── Inference Result ──────────────────────────\n"
            "  Status       : No hemorrhage detected\n"
            f"  Conf. thr.   : {conf_thr:.2f}  |  NMS IoU thr. : {iou_thr:.2f}\n"
            f"  Image shape  : {img_shape[1]}×{img_shape[0]} px\n\n"
            "  All class scores fell below the confidence threshold.\n"
            "  Consider lowering conf. if lesion is subtle or occluded."
        )
    cls_counts = {}
    for name, *_ in dets:
        cls_counts[name] = cls_counts.get(name, 0) + 1

    lines = [
        "── Inference Result ──────────────────────────",
        f"  Detections   : {len(dets)} bounding box(es)   |  Classes: {len(cls_counts)}",
        f"  Conf. thr.   : {conf_thr:.2f}  |  NMS IoU thr. : {iou_thr:.2f}",
        f"  Image shape  : {img_shape[1]}×{img_shape[0]} px",
        "",
        "── Detected Lesions ──────────────────────────",
    ]
    for i, (name, score, x1, y1, x2, y2, wb, hb) in enumerate(dets, 1):
        area_pct = 100.0 * wb * hb / (img_shape[0] * img_shape[1])
        lines += [
            f"  [{i}] {name} — {CLASS_FULL[name]}",
            f"       Anatomy    : {CLASS_LOC[name]}",
            f"       Confidence : {score:.4f}   ({score*100:.1f}%)",
            f"       BBox (xyxy): [{x1}, {y1}, {x2}, {y2}]",
            f"       Size       : {wb}×{hb} px  ({area_pct:.1f}% of frame)",
            "",
        ]
    lines += ["── Class Summary ─────────────────────────────"]
    for name, cnt in sorted(cls_counts.items()):
        lines.append(f"  {name} ({CLASS_FULL[name]}): {cnt} instance(s)")
    lines += [
        "",
        "── Model Info ────────────────────────────────",
        "  Architecture : YOLOv8n (nano)  |  Input: 640×640",
        "  Training     : Roboflow Head CT dataset, 5-class ICH",
        "  Val mAP@0.5  : 0.4721  |  Test mAP@0.5 : 0.6618",
    ]
    return "\n".join(lines)


def build_simple(dets):
    if not dets:
        return (
            "🟢  No hemorrhage detected\n\n"
            "The model found no signs of intracranial hemorrhage in this image.\n"
            "This is not a diagnostic tool; always consult a qualified radiologist for clinical interpretation."
        )
    cls_counts = {}
    for name, *_ in dets:
        cls_counts[name] = cls_counts.get(name, 0) + 1

    lines = [f"🔴  {len(dets)} hemorrhage region(s) detected\n"]
    for name, cnt in sorted(cls_counts.items()):
        lines.append(f"▸ {name} — {cnt} region(s)")
        lines.append(f"  {SIMPLE_DESC[name]}")
        lines.append("")
    lines.append(
        "⚠️  This system is a decision-support tool only. "
        "Clinical diagnosis must be confirmed by a licensed radiologist."
    )
    return "\n".join(lines)


# ── slider row ───────────────────────────────────────────────────────────────

class SliderRow(QWidget):
    def __init__(self, label, mn, mx, default, step, decimals=2):
        super().__init__()
        self._mn = mn
        self._step = step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(mkl(label, color=FG_DIM, size=12))
        orient = Qt.Horizontal if QT_VER == 5 else Qt.Orientation.Horizontal
        self.slider = QSlider(orient)
        self.slider.setRange(0, round((mx - mn) / step))
        self.slider.setValue(round((default - mn) / step))
        exp = QSizePolicy.Expanding if QT_VER == 5 else QSizePolicy.Policy.Expanding
        fix = QSizePolicy.Fixed if QT_VER == 5 else QSizePolicy.Policy.Fixed
        self.slider.setSizePolicy(exp, fix)
        lay.addWidget(self.slider)
        self.val_lbl = QLabel(f"{default:.{decimals}f}")
        self.val_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold; background:transparent;")
        self.val_lbl.setFixedWidth(38)
        align = (Qt.AlignRight | Qt.AlignVCenter) if QT_VER == 5 else (Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val_lbl.setAlignment(align)
        lay.addWidget(self.val_lbl)
        self._dec = decimals
        self.slider.valueChanged.connect(
            lambda: self.val_lbl.setText(f"{self.value():.{self._dec}f}")
        )

    def value(self):
        return self._mn + self.slider.value() * self._step


# ── image label ───────────────────────────────────────────────────────────────

class ImageLabel(QLabel):
    def __init__(self, placeholder=""):
        super().__init__(placeholder)
        align = Qt.AlignCenter if QT_VER == 5 else Qt.AlignmentFlag.AlignCenter
        self.setAlignment(align)
        self.setStyleSheet(
            f"background:{BG2}; color:{FG_DIM}; border:1px solid {BORDER}; border-radius:6px;"
        )
        exp = QSizePolicy.Expanding if QT_VER == 5 else QSizePolicy.Policy.Expanding
        self.setSizePolicy(exp, exp)
        self.setMinimumSize(300, 260)
        self._pix = None

    def set_image(self, arr_rgb):
        self._pix = numpy_to_pixmap(arr_rgb)
        self._rescale()

    def _rescale(self):
        if self._pix:
            keep = Qt.KeepAspectRatio if QT_VER == 5 else Qt.AspectRatioMode.KeepAspectRatio
            smooth = Qt.SmoothTransformation if QT_VER == 5 else Qt.TransformationMode.SmoothTransformation
            self.setPixmap(self._pix.scaled(self.size(), keep, smooth))

    def resizeEvent(self, e):
        self._rescale()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Hemorrhage Detection — YOLOv8n  |  CE 480 Deep Learning")
        self.resize(1280, 780)
        self._img = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(14)

        # ── LEFT panel ────────────────────────────────────────────────────────
        left_w = QWidget()
        left_w.setFixedWidth(260)
        left = QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)

        btn_open = QPushButton("Open CT Image…")
        btn_open.clicked.connect(self._open)
        left.addWidget(btn_open)

        sep = QFrame()
        hline_shape = QFrame.HLine if QT_VER == 5 else QFrame.Shape.HLine
        sep.setFrameShape(hline_shape)
        sep.setStyleSheet(f"color:{BORDER};")
        left.addWidget(sep)

        left.addWidget(mkl("Detection Parameters", bold=True, color=FG_DIM, size=11))
        self.conf = SliderRow("Confidence Thr.", 0.01, 0.95, 0.10, 0.01)
        self.iou  = SliderRow("NMS IoU Thr.",    0.10, 0.90, 0.50, 0.05)
        left.addWidget(self.conf)
        left.addWidget(self.iou)

        btn_run = QPushButton("▶  Run Detection")
        btn_run.clicked.connect(self._run)
        left.addWidget(btn_run)
        left.addStretch()

        # ── RIGHT panel ───────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        img_row = QHBoxLayout()
        img_row.setSpacing(12)

        inp_col = QVBoxLayout()
        inp_col.setSpacing(4)
        inp_col.addWidget(mkl("Input CT Scan", bold=True, color=FG_DIM, size=11))
        self.img_in = ImageLabel("Open a CT image to begin")
        inp_col.addWidget(self.img_in, 1)

        out_col = QVBoxLayout()
        out_col.setSpacing(4)
        out_col.addWidget(mkl("Detection Output", bold=True, color=FG_DIM, size=11))
        self.img_out = ImageLabel("Detection result will appear here")
        out_col.addWidget(self.img_out, 1)

        img_row.addLayout(inp_col, 1)
        img_row.addLayout(out_col, 1)
        right.addLayout(img_row, 3)

        report_row = QHBoxLayout()
        report_row.setSpacing(12)

        tech_col = QVBoxLayout()
        tech_col.setSpacing(4)
        tech_col.addWidget(mkl("Technical Report", bold=True, color=FG_DIM, size=11))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        mono_font = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 11)
        self.summary.setFont(mono_font)
        self.summary.setPlaceholderText(
            "Open an image and press ▶ Run Detection to see technical output…"
        )
        tech_col.addWidget(self.summary)

        simple_col = QVBoxLayout()
        simple_col.setSpacing(4)
        simple_col.addWidget(mkl("Plain Language Summary", bold=True, color=FG_DIM, size=11))
        self.simple = QTextEdit()
        self.simple.setReadOnly(True)
        sans_font = QFont("Helvetica Neue" if sys.platform == "darwin" else "Segoe UI", 12)
        self.simple.setFont(sans_font)
        self.simple.setPlaceholderText("Plain language result will appear here…")
        simple_col.addWidget(self.simple)

        report_row.addLayout(tech_col, 3)
        report_row.addLayout(simple_col, 2)
        right.addLayout(report_row, 2)

        main.addWidget(left_w)
        main.addLayout(right, 1)

    def load_image_path(self, p):
        self._img = np.array(Image.open(p).convert("RGB"))
        self.img_in.set_image(self._img)
        self.img_out.setText("Press ▶ Run Detection")
        self.summary.clear()
        self.simple.clear()

    def _open(self):
        samples = os.path.join(ROOT_DIR, "samples")
        p, _ = QFileDialog.getOpenFileName(
            self, "Open CT Image", samples,
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if p:
            self.load_image_path(p)

    def _run(self):
        if self._img is None:
            self.summary.setText("No image loaded. Please open a CT scan first.")
            return
        conf = self.conf.value()
        iou = self.iou.value()
        out, dets = run_detect(self._img, conf, iou)
        self.img_out.set_image(out)
        self.summary.setText(build_technical(dets, conf, iou, self._img.shape))
        self.simple.setText(build_simple(dets))


def launch_gui():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    launch_gui()
