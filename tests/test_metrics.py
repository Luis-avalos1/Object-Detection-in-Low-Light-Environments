"""Known-answer tests for the COCO-style evaluator and IoU."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lowlight.metrics import DetectionEvaluator, ImageEval, iou_matrix  # noqa: E402


def _ev(det_b, det_s, det_l, gt_b, gt_l, cids=(0,), names=None):
    ev = DetectionEvaluator(class_ids=list(cids), class_names=names or {c: str(c) for c in cids})
    ev.add(ImageEval(np.array(det_b, float).reshape(-1, 4), np.array(det_s, float),
                     np.array(det_l, int), np.array(gt_b, float).reshape(-1, 4), np.array(gt_l, int)))
    return ev.compute()


def test_iou_identical_and_partial():
    a = np.array([[0, 0, 10, 10]], float)
    assert abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-6
    b = np.array([[5, 0, 15, 10]], float)
    assert abs(iou_matrix(a, b)[0, 0] - (50 / 150)) < 1e-4  # 0.333


def test_perfect_detection_ap_is_one():
    m = _ev([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [0])
    assert abs(m["per_class"]["0"]["AP50"] - 1.0) < 1e-6
    assert abs(m["recall@0.25"] - 1.0) < 1e-6


def test_one_tp_one_fp_precision_half():
    m = _ev([[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.8], [0, 0],
            [[0, 0, 10, 10]], [0])
    assert abs(m["precision@0.25"] - 0.5) < 1e-6
    assert abs(m["recall@0.25"] - 1.0) < 1e-6
    assert abs(m["per_class"]["0"]["AP50"] - 1.0) < 1e-6  # TP ranked first


def test_missed_gt_is_false_negative():
    # one of two GT detected -> recall 0.5, AP ~0.505 (101-pt)
    m = _ev([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0])
    assert abs(m["recall@0.25"] - 0.5) < 1e-6
    assert 0.49 < m["per_class"]["0"]["AP50"] < 0.51


def test_wrong_class_is_false_positive():
    m = _ev([[0, 0, 10, 10]], [0.9], [1], [[0, 0, 10, 10]], [0],
            cids=(0, 1), names={0: "a", 1: "b"})
    assert m["mAP@0.5"] == 0.0


def test_absent_class_excluded_from_map():
    # class 1 has no GT and no det -> must not drag mAP to 0
    m = _ev([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [0],
            cids=(0, 1), names={0: "a", 1: "b"})
    assert abs(m["mAP@0.5"] - 1.0) < 1e-6
    assert m["per_class"]["b"]["AP50"] is None


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
