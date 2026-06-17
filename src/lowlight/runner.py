"""Benchmark orchestration: run a (detector x enhancement) cell and evaluate.

A "cell" pairs one detector with one enhancement method over a list of
samples. The runner applies the enhancement, runs detection at a low
confidence (so the PR curve is complete), maps detections into the shared
ExDark label space, and feeds the evaluator. It returns the full metric suite
plus latency and the per-image evaluations (kept so we can bootstrap CIs).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from . import detectors as Det
from . import enhancement as Enh
from .dataset import Sample
from .metrics import DetectionEvaluator, ImageEval, coco_dets_to_exdark


@dataclass
class CellResult:
    detector: str
    enhancement: str
    metrics: dict
    sec_per_img: float
    fps: float
    n_images: int
    per_image: list[ImageEval] = field(default_factory=list)


def run_cell(
    model,
    detector_name: str,
    enhancement_name: str,
    samples: list[Sample],
    *,
    device: str | None = None,
    imgsz: int = C.DEFAULT_IMGSZ,
    conf: float = C.DEFAULT_CONF,
    coco_space: bool | None = None,
    keep_per_image: bool = True,
    progress: bool = False,
) -> CellResult:
    device = device or C.pick_device()
    if coco_space is None:
        coco_space = Det.is_coco_model(model)
    enhance = Enh.get(enhancement_name)
    evaluator = DetectionEvaluator()
    per_image: list[ImageEval] = []

    t0 = time.time()
    n = 0
    it = samples
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(samples, desc=f"{detector_name}/{enhancement_name}", leave=False)
        except Exception:
            pass

    for s in it:
        image = cv2.imread(s.image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        if enhancement_name != "original":
            try:
                image = enhance(image)
            except Exception:
                pass
        det = Det.predict(model, image, imgsz=imgsz, conf=conf, device=device)
        if coco_space:
            boxes, scores, labels = coco_dets_to_exdark(det.boxes, det.scores, det.labels)
        else:
            boxes, scores, labels = det.boxes, det.scores, det.labels
        gt = s.load_objects(clip_to=(w, h))
        gt_boxes = np.array([[o.x1, o.y1, o.x2, o.y2] for o in gt], np.float32) if gt else np.zeros((0, 4), np.float32)
        gt_labels = np.array([o.exdark_id for o in gt], int) if gt else np.zeros(0, int)
        ev = ImageEval(boxes, scores, labels, gt_boxes, gt_labels)
        evaluator.add(ev)
        if keep_per_image:
            per_image.append(ev)
        n += 1

    elapsed = time.time() - t0
    spi = elapsed / max(n, 1)
    return CellResult(
        detector=detector_name,
        enhancement=enhancement_name,
        metrics=evaluator.compute(),
        sec_per_img=spi,
        fps=(1.0 / spi if spi > 0 else 0.0),
        n_images=n,
        per_image=per_image,
    )


def bootstrap_map50_ci(per_image: list[ImageEval], n_boot: int = 200, seed: int = C.DEFAULT_SEED) -> tuple[float, float]:
    """Paired bootstrap 95% CI of mAP@0.5 by resampling images."""
    if not per_image:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per_image))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        ev = DetectionEvaluator()
        for i in pick:
            ev.add(per_image[i])
        vals.append(ev.compute()["mAP@0.5"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))
