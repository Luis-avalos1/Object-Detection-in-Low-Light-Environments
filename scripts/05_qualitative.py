#!/usr/bin/env python3
"""Qualitative panel: GT (green) vs predicted (orange) boxes on sample images.

Usage:
    python scripts/05_qualitative.py --detector yolov8n --n 6 [--weights runs/.../best.pt]
"""
import argparse

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from lowlight import config as C
from lowlight import dataset as D
from lowlight import detectors as Det
from lowlight import viz
from lowlight.metrics import coco_dets_to_exdark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="yolov8n")
    ap.add_argument("--weights", default=None, help="fine-tuned best.pt; if set, native ExDark labels")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="display confidence threshold (0.001 eval threshold floods the panel)")
    ap.add_argument("--out", default=str(C.FIGURES_DIR / "qualitative_panel.png"))
    args = ap.parse_args()

    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
    # one representative image from each of several classes
    picks, seen = [], set()
    for s in D.stratified_subset(split["test"], per_class=4):
        if s.folder_class not in seen:
            picks.append(s); seen.add(s.folder_class)
        if len(picks) >= args.n:
            break

    weights = args.weights or args.detector
    model = Det.load_detector(weights)
    coco_space = Det.is_coco_model(model)
    names = (dict(C.EXDARK_ID_TO_CLASS) if not coco_space else C.EXDARK_ID_TO_CLASS)

    panels = []
    for s in picks:
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        det = Det.predict(model, img, conf=args.conf, device=args.device)
        if coco_space:
            b, sc, lb = coco_dets_to_exdark(det.boxes, det.scores, det.labels)
        else:
            b, sc, lb = det.boxes, det.scores, det.labels
        gt = s.load_objects(clip_to=(w, h))
        gtb = [[o.x1, o.y1, o.x2, o.y2] for o in gt]
        gtl = [o.exdark_id for o in gt]
        drawn = viz.draw_boxes(img, gtb, gtl, b, sc, lb, dict(C.EXDARK_ID_TO_CLASS))
        panels.append(viz.label_strip(drawn, f"{s.folder_class}  (GT=green, pred=orange)"))

    grid = viz.tile(panels, cols=3)
    cv2.imwrite(args.out, grid)
    print("Wrote", args.out, grid.shape)


if __name__ == "__main__":
    main()
