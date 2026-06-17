#!/usr/bin/env python3
"""Build the deterministic ExDark split + benchmark subset, and export YOLO format.

Usage:
    python scripts/00_make_split.py [--no-export]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from lowlight import config as C
from lowlight import dataset as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true", help="skip YOLO-format export")
    ap.add_argument("--subset-per-class", type=int, default=40)
    args = ap.parse_args()

    samples = D.build_index()
    print(f"Indexed {len(samples)} image/annotation pairs")
    if len(samples) != 7363:
        print(f"WARNING: expected 7363, got {len(samples)}")

    split = D.make_split(samples)
    splits_dir = C.DATA_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    D.save_split(split, splits_dir / "exdark_split.json")
    for name, st in D.split_stats(split).items():
        print(f"  {name}: {st['images']} images across {len(st['by_class'])} classes")

    # CSV manifest for human/VCS inspection
    import csv
    with open(splits_dir / "exdark_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "image", "annot"])
        for sname, items in split.items():
            for s in items:
                w.writerow([sname, s.folder_class, s.image_path, s.annot_path])

    subset = D.stratified_subset(split["test"], per_class=args.subset_per_class)
    D.save_split({"subset": subset}, splits_dir / f"test_subset{args.subset_per_class}.json")
    print(f"  benchmark subset: {len(subset)} images ({args.subset_per_class}/class)")

    if not args.no_export:
        yaml = D.export_yolo(split)
        print(f"Exported YOLO dataset -> {yaml}")


if __name__ == "__main__":
    main()
