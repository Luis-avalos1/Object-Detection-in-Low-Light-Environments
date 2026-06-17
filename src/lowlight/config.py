"""Central configuration: paths, class taxonomy, and ExDark->COCO mapping.

This module is the single source of truth for the dataset taxonomy. Two
correctness bugs in the original project are fixed here and documented:

  1. The ``Cup`` class existed on disk but was missing from the class map, so
     every Cup ground-truth object was silently dropped from evaluation.
  2. ``Boat`` was mapped to COCO index 9 ("traffic light") instead of 8
     ("boat"), so Boat detections could never match ground truth and Boat AP
     was structurally ~0.

Both are corrected in ``EXDARK_TO_COCO`` below and verified against the
``ultralytics`` COCO ``names`` table.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "ExDark_Dataset"
ANNOTATIONS_DIR = DATA_DIR / "ground_truths"
MODELS_DIR = DATA_DIR / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
ENHANCED_DIR = RESULTS_DIR / "enhanced_images"
DETECTION_DIR = RESULTS_DIR / "detection_results"

# YOLO-format export + fine-tuning artifacts
YOLO_DATASET_DIR = DATA_DIR / "exdark_yolo"
RUNS_DIR = PROJECT_ROOT / "runs"

for _d in (METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
# The 12 ExDark classes. The on-disk folder names match these exactly and are
# also the first token of every ground-truth line (verified across all classes).
EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
EXDARK_CLASS_TO_ID = {c: i for i, c in enumerate(EXDARK_CLASSES)}
EXDARK_ID_TO_CLASS = {i: c for i, c in enumerate(EXDARK_CLASSES)}

# ExDark class -> COCO 80-class index (ultralytics indexing). Used only when
# evaluating a COCO-pretrained detector zero-shot. After fine-tuning we predict
# the 12 native ExDark classes directly and this mapping is not needed.
#
# Verified: ultralytics names {0:person, 1:bicycle, 2:car, 3:motorcycle,
# 5:bus, 8:boat, 15:cat, 16:dog, 39:bottle, 41:cup, 56:chair, 60:dining table}.
EXDARK_TO_COCO = {
    "Bicycle": 1,    # bicycle
    "Boat": 8,       # boat            (was 9 = "traffic light" -> BUG, fixed)
    "Bottle": 39,    # bottle
    "Bus": 5,        # bus
    "Car": 2,        # car
    "Cat": 15,       # cat
    "Chair": 56,     # chair
    "Cup": 41,       # cup             (was missing -> BUG, fixed)
    "Dog": 16,       # dog
    "Motorbike": 3,  # motorcycle
    "People": 0,     # person
    "Table": 60,     # dining table
}

# Reverse: COCO index -> ExDark class. Note COCO 'wine glass' (40) is folded
# into Bottle and 'couch' (57) is NOT counted as Table, to stay conservative.
COCO_TO_EXDARK = {v: k for k, v in EXDARK_TO_COCO.items()}

# The set of COCO ids a zero-shot detector may emit that we care about. Any
# detection whose class is outside this set is ignored during COCO-mode eval.
RELEVANT_COCO_IDS = set(EXDARK_TO_COCO.values())

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".JPEG", ".JPG", ".PNG"}
DEFAULT_SEED = 42
DEFAULT_SPLIT = (0.70, 0.15, 0.15)  # train / val / test, stratified by class
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.001   # low conf at inference; thresholding happens in metrics
DEFAULT_IOU_NMS = 0.6  # NMS IoU for prediction


def pick_device() -> str:
    """Return the best available torch device string for this machine."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
