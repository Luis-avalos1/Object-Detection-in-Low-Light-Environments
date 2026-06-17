"""Qualitative visualisation: draw GT + predictions and tile sample panels."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as C

GT_COLOR = (0, 200, 0)      # green  (ground truth)
PRED_COLOR = (0, 90, 255)   # orange (prediction)


def draw_boxes(image_bgr, gt_boxes, gt_labels, det_boxes, det_scores, det_labels, names):
    img = image_bgr.copy()
    for box, lab in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
    for box, sc, lab in zip(det_boxes, det_scores, det_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        txt = f"{names.get(int(lab), lab)} {sc:.2f}"
        cv2.putText(img, txt, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 2)
    return img


def label_strip(img, text, h=30):
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, img])


def tile(images, cols=3, cell_w=420):
    """Resize images to a common width and tile into a grid."""
    resized = []
    for im in images:
        h, w = im.shape[:2]
        nh = int(h * cell_w / w)
        resized.append(cv2.resize(im, (cell_w, nh)))
    max_h = max(im.shape[0] for im in resized)
    resized = [np.pad(im, ((0, max_h - im.shape[0]), (0, 0), (0, 0))) for im in resized]
    rows = []
    for i in range(0, len(resized), cols):
        row = resized[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
