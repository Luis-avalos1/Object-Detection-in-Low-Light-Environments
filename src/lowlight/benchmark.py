"""High-level benchmark orchestration writing a single ``grid.json``.

Three axes are supported and tagged so the report can separate them:
  * ``axis="detector"``    — zero-shot detector sweep (enhancement fixed).
  * ``axis="enhancement"`` — enhancement sweep on a fixed detector.
The ``regime`` field distinguishes ``frozen-coco`` from ``finetuned`` cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import detectors as Det
from . import runner as R
from .dataset import Sample


def image_entropy_of_enhancer(samples: list[Sample], enhancer_name: str, max_imgs: int = 60) -> float:
    """Mean Shannon entropy of enhancer outputs — a cheap no-reference
    perceptual-quality proxy used to illustrate quality-vs-detection decoupling."""
    from .enhancement import get
    fn = get(enhancer_name)
    vals = []
    for s in samples[:max_imgs]:
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        try:
            out = fn(img) if enhancer_name != "original" else img
        except Exception:
            out = img
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        vals.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def _cell_dict(res: R.CellResult, axis: str, regime: str, ci=None, quality=None) -> dict:
    return {
        "axis": axis,
        "regime": regime,
        "detector": res.detector,
        "enhancement": res.enhancement,
        "metrics": res.metrics,
        "sec_per_img": res.sec_per_img,
        "fps": res.fps,
        "ci95_map50": ci,
        "quality": ({"entropy": quality} if quality is not None else None),
    }


def run_detector_sweep(detector_names, samples, device="cpu", imgsz=C.DEFAULT_IMGSZ, conf=C.DEFAULT_CONF) -> list[dict]:
    cells = []
    for name in detector_names:
        try:
            model = Det.load_detector(name)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        res = R.run_cell(model, name, "original", samples, device=device, imgsz=imgsz, conf=conf,
                         coco_space=True, keep_per_image=True, progress=True)
        ci = R.bootstrap_map50_ci(res.per_image, n_boot=150)
        cells.append(_cell_dict(res, "detector", "frozen-coco", ci=ci))
        print(f"  {name:10s} mAP@0.5={res.metrics['mAP@0.5']:.3f}  FPS={res.fps:.1f}")
    return cells


def run_enhancement_grid(detector_name, enhancer_names, samples, *, model=None, regime="frozen-coco",
                         coco_space=True, device="cpu", imgsz=C.DEFAULT_IMGSZ, conf=C.DEFAULT_CONF,
                         with_quality=True) -> list[dict]:
    if model is None:
        model = Det.load_detector(detector_name)
    cells = []
    for enh in enhancer_names:
        res = R.run_cell(model, detector_name, enh, samples, device=device, imgsz=imgsz, conf=conf,
                         coco_space=coco_space, keep_per_image=True, progress=True)
        ci = R.bootstrap_map50_ci(res.per_image, n_boot=150)
        q = image_entropy_of_enhancer(samples, enh) if with_quality else None
        cells.append(_cell_dict(res, "enhancement", regime, ci=ci, quality=q))
        print(f"  {enh:18s} mAP@0.5={res.metrics['mAP@0.5']:.3f}  entropy={q:.2f}" if q else
              f"  {enh:18s} mAP@0.5={res.metrics['mAP@0.5']:.3f}")
    return cells


def save_grid(cells: list[dict], device: str, extra_meta: dict | None = None) -> Path:
    path = C.METRICS_DIR / "grid.json"
    meta = {"device": device, "seed": C.DEFAULT_SEED, "imgsz": C.DEFAULT_IMGSZ}
    if extra_meta:
        meta.update(extra_meta)
    payload = {"meta": meta, "cells": cells}
    # Merge with any existing cells (so finetuned arms can be appended later),
    # de-duplicating on (axis, regime, detector, enhancement).
    if path.exists():
        try:
            old = json.load(open(path)).get("cells", [])
            key = lambda c: (c["axis"], c["regime"], c["detector"], c["enhancement"])
            keep = {key(c) for c in cells}
            payload["cells"] = [c for c in old if key(c) not in keep] + cells
        except Exception:
            pass
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
