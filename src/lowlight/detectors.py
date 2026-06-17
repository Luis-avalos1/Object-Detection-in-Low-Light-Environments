"""Thin ultralytics adapter producing detections in a unified format.

A detector is loaded by name (``yolov8n``, ``yolov8s``, ``yolo11n``,
``yolov5nu``, ``rtdetr-l`` or a path to a fine-tuned ``best.pt``). Prediction
returns boxes in **original-image pixel coordinates** (ultralytics handles the
internal letterbox + rescale, so we never squash the image ourselves — a fix
over the original code's manual 640x640 resize).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import config as C

# Let unsupported MPS ops fall back to CPU rather than crash.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_KNOWN = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolov5nu": "yolov5nu.pt",
    "rtdetr-l": "rtdetr-l.pt",
}


@dataclass
class Detections:
    boxes: np.ndarray   # (N,4) xyxy in original pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) detector-native class ids


def resolve_weights(name: str) -> str:
    """Map a short name (or path) to an ultralytics-loadable weights string."""
    if os.path.exists(name):
        return name
    return _KNOWN.get(name, name if name.endswith(".pt") else f"{name}.pt")


def load_detector(name: str):
    from ultralytics import YOLO, RTDETR
    weights = resolve_weights(name)
    if "rtdetr" in os.path.basename(weights).lower():
        return RTDETR(weights)
    return YOLO(weights)


def predict(
    model,
    image_bgr: np.ndarray,
    imgsz: int = C.DEFAULT_IMGSZ,
    conf: float = C.DEFAULT_CONF,
    iou: float = C.DEFAULT_IOU_NMS,
    device: str | None = None,
    max_det: int = 300,
) -> Detections:
    """Run the detector on a single BGR image.

    ultralytics accepts BGR numpy arrays directly and returns boxes already
    mapped back to the input image's pixel space.
    """
    device = device or C.pick_device()
    res = model.predict(
        image_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device,
        verbose=False, max_det=max_det,
    )[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))
    return Detections(
        b.xyxy.cpu().numpy().astype(np.float32),
        b.conf.cpu().numpy().astype(np.float32),
        b.cls.cpu().numpy().astype(int),
    )


def is_coco_model(model) -> bool:
    """Heuristic: a COCO model has 80 classes named person/car/...; a fine-tuned
    ExDark model has 12 classes."""
    try:
        return len(model.names) >= 80
    except Exception:
        return True
