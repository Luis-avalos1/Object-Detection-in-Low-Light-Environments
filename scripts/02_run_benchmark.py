#!/usr/bin/env python3
"""Run the zero-shot detector sweep and the enhancement grid -> results/metrics/grid.json.

Usage:
    python scripts/02_run_benchmark.py \
        --detectors yolov8n yolov8s yolo11n \
        --enhancers original gamma adaptive_gamma hist_eq clahe_hsv clahe_lab msr msrcr gray_world dcp \
        --enh-detector yolov8s --subset-per-class 40 --device cpu
"""
import argparse

import _bootstrap  # noqa: F401
from lowlight import benchmark as B
from lowlight import config as C
from lowlight import dataset as D
from lowlight import enhancement as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="*", default=["yolov8n", "yolov8s", "yolo11n"])
    ap.add_argument("--enhancers", nargs="*", default=E.DEFAULT_METHODS)
    ap.add_argument("--enh-detector", default="yolov8s")
    ap.add_argument("--subset-per-class", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=C.DEFAULT_IMGSZ)
    ap.add_argument("--split-json", default=str(C.DATA_DIR / "splits" / "exdark_split.json"))
    args = ap.parse_args()

    split = D.load_split(args.split_json)
    samples = D.stratified_subset(split["test"], per_class=args.subset_per_class)
    print(f"Benchmark on {len(samples)} test images ({args.subset_per_class}/class), device={args.device}\n")

    cells = []
    print("== Zero-shot detector sweep ==")
    cells += B.run_detector_sweep(args.detectors, samples, device=args.device, imgsz=args.imgsz)

    print("\n== Enhancement grid (frozen COCO detector) ==")
    cells += B.run_enhancement_grid(args.enh_detector, args.enhancers, samples,
                                    device=args.device, imgsz=args.imgsz)

    path = B.save_grid(cells, device=args.device,
                       extra_meta={"subset_per_class": args.subset_per_class,
                                   "enh_detector": args.enh_detector})
    print(f"\nWrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
