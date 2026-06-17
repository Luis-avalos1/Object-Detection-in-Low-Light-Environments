"""ExDark dataset: parsing, indexing, reproducible splits, and YOLO export.

The ExDark dataset files each image under a single class folder (its primary
object) but an image may contain objects of *several* classes. Ground-truth
files use the ``bbGt version=3`` format:

    % bbGt version=3
    Car 13 181 222 190 0 0 0 0 0 0 0
    ...

where the five leading tokens are ``ClassName x y w h`` with ``x, y`` the
top-left corner and ``w, h`` the width/height in absolute pixels.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config as C


@dataclass
class GTObject:
    """A single ground-truth object in native ExDark taxonomy."""
    exdark_id: int          # 0..11 native ExDark class id
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def coco_id(self) -> int:
        return C.EXDARK_TO_COCO[C.EXDARK_ID_TO_CLASS[self.exdark_id]]


@dataclass
class Sample:
    """One image with its annotation path and primary (folder) class."""
    image_path: str
    annot_path: str
    folder_class: str       # the ExDark folder the image is filed under
    stem: str

    def load_objects(self, clip_to: tuple[int, int] | None = None) -> list[GTObject]:
        return parse_bbgt(self.annot_path, clip_to=clip_to)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_bbgt(annot_path: str | Path, clip_to: tuple[int, int] | None = None) -> list[GTObject]:
    """Parse a bbGt annotation file into a list of :class:`GTObject`.

    Args:
        annot_path: path to the ``.txt`` annotation file.
        clip_to: optional ``(width, height)`` to clip boxes into image bounds.
    """
    objs: list[GTObject] = []
    try:
        with open(annot_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return objs

    start = 1 if lines and lines[0].lstrip().startswith("%") else 0
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        if cls not in C.EXDARK_CLASS_TO_ID:
            continue
        try:
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        if clip_to is not None:
            W, H = clip_to
            x1 = min(max(x1, 0.0), W)
            y1 = min(max(y1, 0.0), H)
            x2 = min(max(x2, 0.0), W)
            y2 = min(max(y2, 0.0), H)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
        objs.append(GTObject(C.EXDARK_CLASS_TO_ID[cls], x1, y1, x2, y2))
    return objs


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_index(
    images_dir: str | Path = C.IMAGES_DIR,
    annot_dir: str | Path = C.ANNOTATIONS_DIR,
) -> list[Sample]:
    """Scan the dataset and pair every image with its annotation file."""
    images_dir, annot_dir = Path(images_dir), Path(annot_dir)
    samples: list[Sample] = []
    for cls in sorted(os.listdir(images_dir)):
        cls_img_dir = images_dir / cls
        cls_ann_dir = annot_dir / cls
        if not (cls_img_dir.is_dir() and cls_ann_dir.is_dir()):
            continue
        for fn in sorted(os.listdir(cls_img_dir)):
            ext = os.path.splitext(fn)[1]
            if ext.lower() not in {e.lower() for e in C.IMG_EXTENSIONS}:
                continue
            stem = os.path.splitext(fn)[0]
            # ExDark annotation files are typically named "<image filename>.txt"
            # but some installs use "<stem>.txt". Try both.
            cand = [cls_ann_dir / f"{fn}.txt", cls_ann_dir / f"{stem}.txt"]
            annot = next((p for p in cand if p.exists()), None)
            if annot is None:
                continue
            samples.append(Sample(str(cls_img_dir / fn), str(annot), cls, stem))
    return samples


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_split(
    samples: list[Sample],
    ratios: tuple[float, float, float] = C.DEFAULT_SPLIT,
    seed: int = C.DEFAULT_SEED,
) -> dict[str, list[Sample]]:
    """Deterministic split stratified by folder class.

    ExDark ships no split file in this repo, so we build a reproducible
    stratified split (fixed seed) and persist it so every experiment uses the
    identical partition.
    """
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.folder_class].append(s)

    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    r_train, r_val, _ = ratios
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda s: s.stem)
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(n * r_train))
        n_va = int(round(n * r_val))
        out["train"].extend(group[:n_tr])
        out["val"].extend(group[n_tr:n_tr + n_va])
        out["test"].extend(group[n_tr + n_va:])
    return out


def stratified_subset(samples: list[Sample], per_class: int, seed: int = C.DEFAULT_SEED) -> list[Sample]:
    """Take up to ``per_class`` samples from each folder class (deterministic)."""
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.folder_class].append(s)
    rng = random.Random(seed)
    out: list[Sample] = []
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda s: s.stem)
        rng.shuffle(group)
        out.extend(group[:per_class])
    return out


def save_split(split: dict[str, list[Sample]], path: str | Path) -> None:
    serializable = {
        k: [{"image": s.image_path, "annot": s.annot_path, "class": s.folder_class} for s in v]
        for k, v in split.items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[Sample]]:
    with open(path) as f:
        raw = json.load(f)
    return {
        k: [Sample(d["image"], d["annot"], d["class"], os.path.splitext(os.path.basename(d["image"]))[0]) for d in v]
        for k, v in raw.items()
    }


def split_stats(split: dict[str, list[Sample]]) -> dict:
    stats = {}
    for name, items in split.items():
        stats[name] = {"images": len(items), "by_class": dict(Counter(s.folder_class for s in items))}
    return stats


# --------------------------------------------------------------------------- #
# YOLO-format export (for fine-tuning)
# --------------------------------------------------------------------------- #
def export_yolo(
    split: dict[str, list[Sample]],
    out_dir: str | Path = C.YOLO_DATASET_DIR,
    link: bool = True,
) -> Path:
    """Export the split to an ultralytics-compatible YOLO dataset.

    Layout::

        out_dir/
          images/{train,val,test}/<stem>.<ext>   (symlinks by default)
          labels/{train,val,test}/<stem>.txt      (normalized cx cy w h)
          dataset.yaml

    Labels use the 12 native ExDark class ids (0..11), so a fine-tuned model
    predicts ExDark classes directly.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    for sub in ("images", "labels"):
        for part in ("train", "val", "test"):
            (out_dir / sub / part).mkdir(parents=True, exist_ok=True)

    for part, items in split.items():
        for s in items:
            try:  # PIL reads only the header -> fast size lookup, no full decode
                with Image.open(s.image_path) as im:
                    w, h = im.size
            except Exception:
                continue
            objs = parse_bbgt(s.annot_path, clip_to=(w, h))
            if not objs:
                continue
            ext = os.path.splitext(s.image_path)[1]
            dst_img = out_dir / "images" / part / f"{s.stem}{ext}"
            dst_lbl = out_dir / "labels" / part / f"{s.stem}.txt"
            if not dst_img.exists():
                if link:
                    try:
                        os.symlink(os.path.abspath(s.image_path), dst_img)
                    except (OSError, FileExistsError):
                        import shutil
                        shutil.copy2(s.image_path, dst_img)
                else:
                    import shutil
                    shutil.copy2(s.image_path, dst_img)
            with open(dst_lbl, "w") as f:
                for o in objs:
                    cx = ((o.x1 + o.x2) / 2) / w
                    cy = ((o.y1 + o.y2) / 2) / h
                    bw = (o.x2 - o.x1) / w
                    bh = (o.y2 - o.y1) / h
                    f.write(f"{o.exdark_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_path = out_dir / "dataset.yaml"
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(C.EXDARK_CLASSES))
    yaml_path.write_text(
        f"# ExDark in YOLO format (auto-generated)\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(C.EXDARK_CLASSES)}\n"
        f"names:\n{names_block}\n"
    )
    return yaml_path


if __name__ == "__main__":
    samples = build_index()
    print(f"Indexed {len(samples)} samples")
    split = make_split(samples)
    for name, st in split_stats(split).items():
        print(f"  {name}: {st['images']} images")
