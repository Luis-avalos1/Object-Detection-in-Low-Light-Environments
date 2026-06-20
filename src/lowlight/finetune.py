"""Fine-tune an ultralytics detector on ExDark (the dominant accuracy lever).

The literature is unambiguous: zero-shot COCO models score poorly on ExDark
(~21 mAP@0.5 for YOLOv3) and fine-tuning on ExDark is worth far more than any
enhancement choice (~76 mAP@0.5). This module wraps ultralytics training on
the YOLO-format export and records before/after numbers on the held-out test
split.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import config as C
from . import dataset as D


def ensure_export() -> Path:
    yaml = C.YOLO_DATASET_DIR / "dataset.yaml"
    if not yaml.exists():
        split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
        yaml = D.export_yolo(split)
    return yaml


def finetune(
    base: str = "yolov8n.pt",
    epochs: int = 40,
    imgsz: int = C.DEFAULT_IMGSZ,
    batch: int = 16,
    device: str | None = None,
    name: str | None = None,
    fraction: float = 1.0,
    patience: int = 12,
    workers: int = 8,
    freeze: int | None = None,
    lr0: float | None = None,
    data_yaml: str | Path | None = None,
) -> dict:
    """Fine-tune ``base`` on ExDark and return train/val artifacts paths.

    ``freeze`` freezes the first N layers (the COCO-pretrained backbone for
    YOLOv8 is layers 0-9). Freezing preserves general features and only adapts
    the detection head — the key guard against catastrophic forgetting of rare
    classes when fine-tuning on a small subset on a CPU budget.

    ``data_yaml`` overrides the dataset config (e.g. a class-balanced subset
    export); defaults to the full export from :func:`ensure_export`.
    """
    from ultralytics import YOLO

    device = device or C.pick_device()
    yaml = Path(data_yaml) if data_yaml else ensure_export()
    name = name or f"exdark_{Path(base).stem}_e{epochs}"
    model = YOLO(base)
    train_kwargs = dict(
        data=str(yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=C.DEFAULT_SEED,
        patience=patience,
        fraction=fraction,
        workers=workers,
        project=str(C.RUNS_DIR / "detect"),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    return {"name": name, "save_dir": str(save_dir), "best": str(best), "base": base, "epochs": epochs}


def evaluate_on_test(weights: str, device: str | None = None, imgsz: int = C.DEFAULT_IMGSZ,
                     split: str = "test", data_yaml: str | Path | None = None) -> dict:
    """Run ultralytics val() on a split of the export; returns headline mAPs.

    This uses ultralytics' own (correct, pooled) mAP implementation, giving an
    independent cross-check of our hand-rolled evaluator. ``data_yaml`` selects a
    specific dataset (e.g. an enhanced-domain export); defaults to the standard
    original export.
    """
    from ultralytics import YOLO

    device = device or C.pick_device()
    yaml = Path(data_yaml) if data_yaml else ensure_export()
    model = YOLO(weights)
    metrics = model.val(data=str(yaml), split=split, imgsz=imgsz, device=device, verbose=False, plots=False)
    per_class = {}
    try:
        names = model.names
        for i, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
            per_class[names[int(i)]] = float(ap50)
    except Exception:
        pass
    return {
        "weights": weights,
        "split": split,
        "mAP@0.5": float(metrics.box.map50),
        "mAP@[.5:.95]": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_AP50": per_class,
    }


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
