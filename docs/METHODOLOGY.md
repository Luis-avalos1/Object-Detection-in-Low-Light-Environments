# Methodology

This document records *why* the original experiment produced “barely any
improvement,” the bugs that made the result uninterpretable, the fixes, and the
literature that frames the corrected results.

## 1. The original setup (and why it could not show a difference)

The original pipeline (preserved verbatim under
[`legacy/object_detection.py`](../legacy/object_detection.py) and
[`legacy/enhance.py`](../legacy/enhance.py)) loaded a **COCO-pretrained, frozen**
YOLOv5, applied a set of classical enhancements (histogram equalisation, CLAHE,
single/multi-scale Retinex, gamma), ran detection on each, and reported a single
Precision / Recall / AP triple per method.

Three independent flaws each, on their own, flatten or invalidate the comparison:

### 1.1 The metric was not mAP
`calculate_average_metrics()` took the **last element** of each image’s
cumulative precision and recall arrays and averaged those across all ~7,363
images:

```python
avg_p  = np.mean([m['precision'][-1] for m in metrics])   # per-image terminal point
avg_re = np.mean([m['recall'][-1]    for m in metrics])
```

`precision[-1]`/`recall[-1]` collapse the **entire precision–recall curve** to a
single operating point, so a method that ranks true detections more confidently
than false ones (exactly what AP rewards) gets no credit. Per-image averaging
also weights a 1-object image the same as a 20-object image, and the many sparse
ExDark images dominate the mean. Net effect: the statistic has almost no
resolution for the effect being measured — methods look near-identical regardless
of whether enhancement helps.

`average_precision_score(matches, scores)` (sklearn) compounded this: its samples
are only the detector’s own predictions, so a **missed** ground-truth object is
invisible to it — false negatives can never lower the score. That is backwards
from detection AP, where the recall denominator must be the total ground truth.

### 1.2 The class map silently corrupted two classes
```python
coco_mapping = {... 'Boat': 9, ...}   # 9 == COCO "traffic light", not boat (8)
# 'Cup' absent entirely  -> every Cup object dropped
```
- **Boat → 9** points at *traffic light*; Boat detections could never match Boat
  ground truth, so Boat AP was structurally ~0.
- **Cup** was missing from the map, so all Cup objects were dropped before
  evaluation.

Two of twelve classes were therefore broken before any enhancement ran.

### 1.3 The comparison ran on a frozen, out-of-domain detector
The COCO-pretrained detector was never adapted to ExDark. Worse, Retinex/MSR
output is a normalised **log-difference image** — grey, halo’d, hue-shifted — that
looks nothing like the natural photographs the network’s filters and BatchNorm
statistics were trained on. Feeding it to a frozen COCO model is a large
covariate shift, so Retinex/MSR almost always *underperform* the original. That
is a confound, not a result: the experiment never tested those enhancements
fairly.

## 2. The fixes

| Area | Fix | Where |
|---|---|---|
| Metric | Pooled, COCO-style evaluator: per-class greedy IoU matching over all detections, recall vs **total GT**, 101-point AP, mAP@0.5 and mAP@[.5:.95] | `src/lowlight/metrics.py` |
| Taxonomy | `Boat→8`, `Cup→41` added; native 12-class head used for fine-tuning | `src/lowlight/config.py` |
| Domain | Natural-appearance MSRCR (colour restoration + percentile clip) instead of raw log-difference; correct experimental order (fine-tune first) | `src/lowlight/enhancement.py` |
| Resize | Let ultralytics letterbox internally; boxes mapped back to original pixels (no 640×640 squash) | `src/lowlight/detectors.py` |
| Evaluator validation | Known-answer unit tests + cross-check vs ultralytics `val()` | `tests/`, `src/lowlight/finetune.py` |

## 3. What the literature says (so results are interpretable)

- **Classical enhancement rarely improves downstream detection.** Perceptual
  quality (PSNR/SSIM/NIQE) is decoupled from detection mAP; enhancement also
  amplifies sensor noise in exactly the dark regions where objects live. Net mAP
  change is typically within ±1–2 points and can be negative
  (e.g. arXiv 2311.18814 on underwater; MDPI Electronics 12(16):3517).
- **Fine-tuning the detector on ExDark is the dominant lever.** Zero-shot
  COCO YOLOv3 scores ~21 mAP@0.5; fine-tuned on ExDark it reaches ~76. Every
  illumination-aware method (MAET 77.7, IAT 77.8, PE-YOLO 78.0) sits within
  ~1.6 mAP of a plain fine-tuned baseline — strong evidence that, once the
  detector is fine-tuned, fancy enhancement adds little.
- **Correct experimental order:** record the zero-shot baseline → fine-tune the
  detector (the big gain) → only then test enhancement arms, ideally with the
  detector adapted to the enhanced domain.

### Key references
- MAET, *Multitask AET with Orthogonal Tangent Regularity* (ICCV 2021) — github.com/cuiziteng/ICCV_MAET
- IAT, *Illumination Adaptive Transformer* (BMVC 2022) — github.com/cuiziteng/Illumination-Adaptive-Transformer
- PE-YOLO (BMVC 2023) — arXiv:2307.10953
- Zero-DCE (CVPR 2020) — learned, zero-reference low-light enhancement
- *Two-stage object detection in low-light using deep image enhancement* — PMC12190514
- ExDark dataset — github.com/cs-chan/Exclusively-Dark-Image-Dataset

## 4. Compute note (honesty about scale)

All experiments here run on an Apple M1 (CPU/MPS, no CUDA). **Inference** runs on
MPS/CPU. **Training** runs on **CPU** because ultralytics 8.3.27 + torch 2.5.1
hit a shape-mismatch bug in the task-aligned assigner on the MPS backend. CPU
training of the full 5,153-image train split is ~1 hour/epoch, so the fine-tune
shipped here is a **scoped demonstration** (subset + reduced epochs) that
establishes the before/after direction on real data. The exact recipe to
reproduce the full ~76 mAP@0.5 result on a CUDA machine is in
`scripts/03_finetune.py` (`--fraction 1.0 --epochs 100 --device 0`).
