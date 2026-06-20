# Object Detection in Low-Light Environments

A reproducible study of **image enhancement** and **detector fine-tuning** for
object detection on the [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
low-light dataset — rebuilt from a course project into a research-grade
benchmark with a **correct COCO-style mAP evaluator**, multiple detectors, and an
honest ablation.

> **TL;DR.** The original project concluded that low-light enhancement gave
> *“barely any improvement.”* This rebuild shows *why*: the evaluation metric
> could not have detected a difference, two of the twelve classes were silently
> broken, and enhancement was being tested on a **frozen** COCO detector that was
> never adapted to the enhanced domain. After fixing the evaluator and the
> taxonomy and **fine-tuning the detector on ExDark**, the corrected picture
> matches the published consensus: **fine-tuning is the dominant lever; classical
> enhancement on a fixed detector is roughly neutral.**

📄 **Full results:** [`results/report.md`](results/report.md) ·
🧪 **Methodology & bug analysis:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

---

## Why this is interesting

The original code reported one Precision/Recall/AP triple per enhancement method
and they were all nearly identical. That was taken as “enhancement doesn’t help.”
In fact three separate flaws made the experiment uninterpretable:

| Flaw in the original code | Consequence | Fix |
|---|---|---|
| Averaged per-image `precision[-1]` / `recall[-1]` as “mAP” | Collapses the PR curve; signal washed out; numbers non-comparable | Pooled COCO-style evaluator (`metrics.py`): 101-pt AP, mAP@0.5 & mAP@[.5:.95] |
| `average_precision_score(match, score)` per image | Missed objects (false negatives) invisible | Recall denominator = total GT per class across the dataset |
| `Cup` missing from the class map | All Cup objects dropped | `Cup → COCO 41` + native 12-class head |
| `Boat` mapped to COCO 9 (*traffic light*) | Boat AP structurally ≈ 0 | `Boat → COCO 8` |
| Raw Retinex/MSR log-images → frozen COCO model | Out-of-distribution; unfair test | Natural-appearance **MSRCR**; fine-tune first |
| Manual 640×640 squash resize | Aspect-ratio distortion | Let ultralytics letterbox; boxes mapped back to original pixels |

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full write-up with code
references and literature.

## What’s new

- **Correct metrics** — a pooled, COCO-style evaluator (`src/lowlight/metrics.py`)
  with mAP@0.5, mAP@[.5:.95], per-class AP, and bootstrap confidence intervals;
  cross-checked against ultralytics `val()` and covered by known-answer unit tests.
- **More models** — YOLOv8 (n/s/m), YOLO11 (n/s), YOLOv5(u), and RT-DETR via a
  single ultralytics adapter (`src/lowlight/detectors.py`).
- **More enhancement methods** — 12 in a registry (`src/lowlight/enhancement.py`):
  gamma / adaptive-gamma, brightness-contrast, HE, CLAHE (HSV & LAB), SSR, MSR,
  **MSRCR**, gray-world white balance, and dark-channel-prior low-light.
- **Fine-tuning** — convert ExDark → YOLO format and fine-tune a native 12-class
  detector (`src/lowlight/finetune.py`), with a documented before/after.
- **Reporting** — figures, CSV tables, and a markdown/HTML research report
  generated from the metrics (`src/lowlight/report.py`).

## Repository layout

```
src/lowlight/
  config.py        # paths, 12-class taxonomy, verified ExDark->COCO map, device
  dataset.py       # bbGt parsing, reproducible stratified split, YOLO export
  enhancement.py   # 12 enhancement methods in a registry
  detectors.py     # ultralytics adapter (YOLOv8/11/v5/RT-DETR) -> unified dets
  metrics.py       # correct pooled COCO-style mAP evaluator
  runner.py        # run a (detector x enhancement) cell + bootstrap CIs
  benchmark.py     # orchestrate the sweep/grid -> results/metrics/grid.json
  finetune.py      # fine-tune on ExDark + evaluate (cross-checks evaluator)
  report.py        # figures + CSV + markdown/HTML report
  viz.py           # qualitative GT-vs-prediction panels
scripts/           # 00_make_split, 02_run_benchmark, 03_finetune, 04_make_report, 05_qualitative
tests/             # known-answer tests for the evaluator and enhancers
docs/METHODOLOGY.md
results/           # metrics (json/csv), figures, report.md / report.html
```

## Quickstart

One cross-platform command runs the whole study and auto-detects the best device
(CUDA → MPS → CPU). Works identically on **macOS, Windows, and Linux** — no shell
scripts, no hardcoded interpreter paths:

```bash
python -m venv venv
# macOS/Linux:  source venv/bin/activate
# Windows:      venv\Scripts\activate
pip install -r requirements.txt

# Place the ExDark dataset under data/ (see "Dataset" below), then:
python run.py --quick          # fast smoke test (5 imgs/class, ~minutes)
python run.py                  # full benchmark across detectors × enhancements
python run.py --full           # 6 detector families × all 12 enhancers
python run.py --full --finetune --device auto   # add the fine-tuning arm (GPU rec.)
```

`run.py` produces a **detector × enhancement** comparison
(`results/comparison.md` + `figures/model_enhancement_matrix.png`) on top of the
standard report. Pick stages and knobs explicitly, too:

```bash
python run.py --stages env split matrix report \
  --detectors yolov8n yolov9t yolov10n yolo11n rtdetr-l \
  --enhancers original clahe_lab adaptive_gamma hist_eq --per-class 25
```

<details><summary>Or run the original per-stage scripts directly</summary>

```bash
python scripts/00_make_split.py        # deterministic split + YOLO export
python scripts/02_run_benchmark.py     # zero-shot sweep + enhancement grid -> results/metrics/grid.json
python scripts/03_finetune.py          # fine-tune + before/after          -> results/metrics/finetune.json
python scripts/04_make_report.py       # figures + tables + report.md/html
python scripts/05_qualitative.py       # GT vs prediction panels
pytest -q                              # evaluator/enhancer unit tests
```
</details>

Benchmark + inference run on CPU/MPS (Apple Silicon) — **no CUDA required**.

### Running on more than one machine

Inference mAP is **device-independent** — the same seed, weights, and images give
the same mAP on Apple MPS and on an NVIDIA GPU; only throughput (FPS) and the
fine-tuning arm differ. So keep `results/` as the single canonical run (the GPU
box is recommended, since fine-tuning needs it) and use `--tag` to keep a
per-machine snapshot under `results/snapshots/<tag>/` for provenance:

```bash
python run.py --tag mac-mps      # on the Mac
python run.py --full --finetune --tag pc-cuda --device auto   # on the Windows GPU box
```

Each snapshot carries a `RUN_INFO.txt` (platform, device, settings). This shows
cross-platform reproducibility (identical mAP, different FPS) without two
divergent result trees.

### Fine-tuning on a GPU machine (recommended)

CPU training is slow, and a small/biased subset causes catastrophic forgetting
(rare classes collapse to ~0 AP). On a CUDA box, fine-tune on the **full** train
split — `--fraction 1.0` uses all 5,153 images so no class is starved:

```bash
make finetune DEVICE=0          # full fine-tune, all data, imgsz 640, 100 epochs
# or directly:
python scripts/03_finetune.py --base yolov8n.pt --fraction 1.0 --epochs 100 --imgsz 640 --device 0
```

On a weak GPU/CPU, use the class-balanced freeze-backbone demo (keeps the COCO
backbone, trains the head on a balanced 120-img/class subset — avoids the
forgetting that a naive small `--fraction` causes):

```bash
make finetune-demo DEVICE=0     # freeze=10, balanced subset, imgsz 416, 24 epochs
```

## Dataset

[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset): 7,363
low-light images across 12 classes (Bicycle, Boat, Bottle, Bus, Car, Cat, Chair,
Cup, Dog, Motorbike, People, Table), with `bbGt` bounding boxes. Expected layout:

```
data/ExDark_Dataset/<Class>/<image>          data/ground_truths/<Class>/<image>.txt
```

A deterministic, class-stratified 70/15/15 split (seed 42) is written to
`data/splits/` so every experiment uses the identical partition.

## Metrics

- **mAP@0.5** — primary, matches ExDark literature.
- **mAP@[.5:.95]** — COCO-strict (AP averaged over IoU 0.50:0.05:0.95).
- **Per-class AP** for all 12 classes (the recovered Cup & Boat columns are now non-zero).
- **Precision / Recall / F1** at a stated operating point (conf 0.25), reported
  separately from AP (never used to truncate the PR curve).
- **Bootstrap 95% CIs** over images, to quantify whether differences are real.

## Results

The headline figures and tables are generated into `results/` and summarised in
[`results/report.md`](results/report.md):

- `figures/zeroshot_vs_finetuned.png` — fine-tuning is the dominant lever.
- `figures/map_by_enhancement.png` — enhancement effect on a fixed detector.
- `figures/per_class_ap_heatmap.png` — per-class AP (Cup/Boat recovered).
- `figures/accuracy_vs_latency.png` — accuracy/speed across detectors.

Published ExDark context (fine-tuned YOLOv3, official split, mAP@0.5): zero-shot
COCO ~0.21 -> fine-tuned baseline 0.764 -> MAET 0.777 -> IAT 0.778 -> PE-YOLO 0.780.
The entire SOTA spread is ~1.6 mAP, underscoring that the lever is fine-tuning,
not enhancement.

## Acknowledgements

ExDark dataset (Loh & Chan, 2019); Ultralytics YOLO; and the MAET / IAT /
PE-YOLO / Zero-DCE lines of work cited in `docs/METHODOLOGY.md`.
