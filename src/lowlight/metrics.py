"""Correct COCO-style detection metrics.

The original project averaged each image's *last* cumulative precision/recall
value and called it mAP. That is not a valid detection metric: it ignores
missed ground truth (false negatives are invisible because recall is computed
per image with that image's own GT count), pools nothing across the dataset,
and never integrates a precision-recall curve per class. The net effect is a
number that barely moves between methods — exactly the symptom reported.

This module implements a proper evaluator:

  * detections are pooled across the whole dataset, per class;
  * within each class, detections are sorted by descending confidence and
    greedily matched to ground truth by IoU (each GT used at most once);
  * unmatched ground truth counts as a false negative;
  * Average Precision is the area under the (monotonised) precision-recall
    curve using COCO's 101-point interpolation;
  * AP is computed at IoU 0.50 (AP50), 0.75 (AP75), and averaged over
    IoU 0.50:0.05:0.95 (the primary COCO mAP);
  * mAP is the mean of per-class AP over classes that have ground truth.

All boxes are ``[x1, y1, x2, y2]`` in absolute pixels in a single, shared
label space (we use the 12 native ExDark class ids).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C

COCO_IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class ImageEval:
    """Per-image detections and targets in a shared label space."""
    det_boxes: np.ndarray   # (N, 4)
    det_scores: np.ndarray  # (N,)
    det_labels: np.ndarray  # (N,)
    gt_boxes: np.ndarray    # (M, 4)
    gt_labels: np.ndarray   # (M,)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return (inter / union).astype(np.float32)


def _ap_from_pr(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from sorted-by-confidence tp flags."""
    if n_gt == 0:
        return float("nan")  # class not present -> excluded from mAP
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt  # n_gt > 0 guaranteed above
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Monotonic (precision envelope) then sample at 101 recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in RECALL_POINTS:
        idx = np.searchsorted(recall, r, side="left")
        p = precision[idx] if idx < len(precision) else 0.0
        ap += p / len(RECALL_POINTS)
    return float(ap)


class DetectionEvaluator:
    """Accumulate per-image results, then compute COCO-style mAP.

    Works in any single integer label space; here we use ExDark ids 0..11.
    """

    def __init__(self, class_ids: list[int] | None = None, class_names: dict[int, str] | None = None):
        self.class_ids = class_ids if class_ids is not None else list(range(len(C.EXDARK_CLASSES)))
        self.class_names = class_names or dict(C.EXDARK_ID_TO_CLASS)
        # per (class, iou) -> list of (score, tp)
        self._records: dict[tuple[int, float], list[tuple[float, int]]] = {}
        self._n_gt: dict[int, int] = {c: 0 for c in self.class_ids}
        # confusion-style tallies at the primary operating point (filled lazily)
        self._images = 0

    def add(self, ev: ImageEval) -> None:
        self._images += 1
        for c in self.class_ids:
            gt_mask = ev.gt_labels == c
            det_mask = ev.det_labels == c
            gt_boxes = ev.gt_boxes[gt_mask]
            self._n_gt[c] += int(gt_mask.sum())

            det_boxes = ev.det_boxes[det_mask]
            det_scores = ev.det_scores[det_mask]
            if len(det_boxes):
                order = np.argsort(-det_scores)
                det_boxes = det_boxes[order]
                det_scores = det_scores[order]

            ious = iou_matrix(det_boxes, gt_boxes)  # (Nd, Ng)
            for thr in COCO_IOU_THRESHOLDS:
                key = (c, float(round(thr, 2)))
                rec = self._records.setdefault(key, [])
                matched = np.zeros(len(gt_boxes), dtype=bool)
                for di in range(len(det_boxes)):
                    tp = 0
                    if len(gt_boxes):
                        gi = int(np.argmax(ious[di]))
                        if ious[di, gi] >= thr and not matched[gi]:
                            matched[gi] = True
                            tp = 1
                    rec.append((float(det_scores[di]), tp))

    # ----------------------------------------------------------------- #
    def _ap(self, c: int, thr: float) -> float:
        rec = self._records.get((c, float(round(thr, 2))), [])
        if rec:
            scores = np.array([s for s, _ in rec])
            tps = np.array([t for _, t in rec])
        else:
            scores = np.array([])
            tps = np.array([])
        return _ap_from_pr(tps, scores, self._n_gt[c])

    def per_class_ap(self, thr: float = 0.5) -> dict[str, float]:
        return {self.class_names[c]: self._ap(c, thr) for c in self.class_ids}

    def compute(self) -> dict:
        """Return the full metric suite as a nested dict of plain floats."""
        present = [c for c in self.class_ids if self._n_gt[c] > 0]

        ap50 = {c: self._ap(c, 0.5) for c in self.class_ids}
        ap75 = {c: self._ap(c, 0.75) for c in self.class_ids}

        def _mean_over_iou(c):
            vals = [self._ap(c, t) for t in COCO_IOU_THRESHOLDS]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        ap_coco = {c: _mean_over_iou(c) for c in self.class_ids}

        def mean_present(d):
            vals = [d[c] for c in present if not np.isnan(d[c])]
            return float(np.mean(vals)) if vals else 0.0

        per_class = {
            self.class_names[c]: {
                "AP50": _nan_to_none(ap50[c]),
                "AP75": _nan_to_none(ap75[c]),
                "AP@[.5:.95]": _nan_to_none(ap_coco[c]),
                "n_gt": self._n_gt[c],
            }
            for c in self.class_ids
        }

        # Operating-point precision/recall/F1 at conf>=0.25, IoU 0.5
        pr = self._operating_point(conf_thr=0.25, iou_thr=0.5)

        return {
            "mAP@0.5": mean_present(ap50),
            "mAP@0.75": mean_present(ap75),
            "mAP@[.5:.95]": mean_present(ap_coco),
            "precision@0.25": pr["precision"],
            "recall@0.25": pr["recall"],
            "f1@0.25": pr["f1"],
            "n_images": self._images,
            "n_gt_total": int(sum(self._n_gt.values())),
            "per_class": per_class,
        }

    def _operating_point(self, conf_thr: float, iou_thr: float) -> dict:
        """Aggregate precision/recall/F1 across all classes at a fixed conf."""
        tp = fp = 0
        n_gt = sum(self._n_gt[c] for c in self.class_ids)
        for c in self.class_ids:
            rec = self._records.get((c, float(round(iou_thr, 2))), [])
            for score, t in rec:
                if score >= conf_thr:
                    tp += t
                    fp += 1 - t
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (n_gt + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _nan_to_none(x: float):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


# --------------------------------------------------------------------------- #
# Helpers to convert detector outputs into the shared (ExDark) label space
# --------------------------------------------------------------------------- #
def coco_dets_to_exdark(boxes, scores, labels):
    """Map COCO-space detections to ExDark label space, dropping irrelevant classes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    keep = np.array([l in C.COCO_TO_EXDARK for l in labels], dtype=bool)
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    mapped = np.array([C.EXDARK_CLASS_TO_ID[C.COCO_TO_EXDARK[int(l)]] for l in labels[keep]], dtype=int)
    return boxes[keep], scores[keep], mapped
