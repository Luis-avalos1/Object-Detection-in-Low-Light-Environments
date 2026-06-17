#!/usr/bin/env python3
"""Fine-tune a detector on ExDark and record before/after -> results/metrics/finetune.json.

Usage:
    python scripts/03_finetune.py --base yolov8n.pt --epochs 40 --batch 16 --device mps
"""
import argparse
import json

import _bootstrap  # noqa: F401
from lowlight import config as C
from lowlight import dataset as D
from lowlight import detectors as Det
from lowlight import finetune as FT
from lowlight import runner as R


def _write_img_list(samples, part: str, tag: str) -> "tuple[str, int]":
    """Write a newline-separated list of exported image paths for ``samples``."""
    import os
    img_dir = C.YOLO_DATASET_DIR / "images" / part
    listed = []
    for s in samples:
        hits = list(img_dir.glob(f"{s.stem}.*"))  # export keeps original extension
        if hits:
            # abspath (NOT resolve): keep the in-export ``/images/`` path so
            # ultralytics finds the sibling ``/labels/`` tree. resolve() would
            # follow the symlink back to the original and break label lookup.
            listed.append(os.path.abspath(hits[0]))
    list_path = C.YOLO_DATASET_DIR / f"{tag}.txt"
    list_path.write_text("\n".join(listed) + "\n")
    return list_path.name, len(listed)


def build_balanced_yaml(per_class: int, val_per_class: int = 40) -> str:
    """Build a class-balanced train (+val) subset and a dataset.yaml for it.

    ultralytics' ``fraction`` flag keeps the *first* X% of (filename-sorted)
    images, and ExDark filenames are roughly class-contiguous — so a small
    fraction starves late classes. Instead we draw ``per_class`` images from
    every folder class (deterministic) and write the subset as an image-list
    .txt that ultralytics reads as ``train:``. A balanced ``val_per_class``
    subset keeps per-epoch validation cheap on CPU. test stays full.
    """
    FT.ensure_export()  # guarantees images/labels symlinks + base dataset.yaml exist
    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
    tr = D.stratified_subset(split["train"], per_class=per_class)
    va = D.stratified_subset(split["val"], per_class=val_per_class)
    tr_name, n_tr = _write_img_list(tr, "train", f"train_balanced_{per_class}")
    va_name, n_va = _write_img_list(va, "val", f"val_balanced_{val_per_class}")
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(C.EXDARK_CLASSES))
    yaml_path = C.YOLO_DATASET_DIR / f"dataset_balanced_{per_class}.yaml"
    yaml_path.write_text(
        f"# ExDark class-balanced subset ({per_class}/class train, "
        f"{val_per_class}/class val) — auto-generated\n"
        f"path: {C.YOLO_DATASET_DIR.resolve()}\n"
        f"train: {tr_name}\n"
        f"val: {va_name}\n"
        f"test: images/test\n"
        f"nc: {len(C.EXDARK_CLASSES)}\n"
        f"names:\n{names_block}\n"
    )
    print(f"      balanced subset: {n_tr} train / {n_va} val images")
    return str(yaml_path)


def handrolled_subset_map(model, name: str, device: str, coco_space: bool, n_per_class: int = 40) -> dict:
    """Evaluate a model on the test subset with our pooled evaluator.

    Using the SAME evaluator, images, and (ExDark) label space for both the
    zero-shot and fine-tuned models makes the before/after directly comparable.
    """
    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
    sub = D.stratified_subset(split["test"], per_class=n_per_class)
    res = R.run_cell(model, name, "original", sub, device=device, coco_space=coco_space, keep_per_image=False)
    return {"mAP@0.5": res.metrics["mAP@0.5"], "mAP@[.5:.95]": res.metrics["mAP@[.5:.95]"],
            "recall@0.25": res.metrics["recall@0.25"],
            "per_class": {k: v["AP50"] for k, v in res.metrics["per_class"].items()},
            "n_images": res.metrics["n_images"],
            "source": ("hand-rolled evaluator, COCO->ExDark, test subset" if coco_space
                       else "hand-rolled evaluator, native 12-class, test subset")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=C.DEFAULT_IMGSZ)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--eval-device", default="cpu")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze first N layers (10 = YOLOv8 backbone) to avoid catastrophic forgetting")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--data-yaml", default=None,
                    help="override dataset.yaml (e.g. a class-balanced subset export)")
    ap.add_argument("--per-class", type=int, default=None,
                    help="if set, build a class-balanced train subset of this many imgs/class and train on it")
    args = ap.parse_args()

    device = args.device or C.pick_device()
    out_path = C.METRICS_DIR / "finetune.json"
    record = {"models": []}
    if out_path.exists():
        record = json.load(open(out_path))

    data_yaml = args.data_yaml
    if args.per_class:
        data_yaml = build_balanced_yaml(args.per_class)
        print(f"      built class-balanced train subset -> {data_yaml}")

    print(f"[1/4] Zero-shot baseline for {args.base} (hand-rolled, test subset) ...")
    zs_model = Det.load_detector(args.base.replace(".pt", ""))
    zs = handrolled_subset_map(zs_model, args.base, args.eval_device, coco_space=True)
    print(f"      zero-shot mAP@0.5 = {zs['mAP@0.5']:.3f}  mAP@.5:.95 = {zs['mAP@[.5:.95]']:.3f}")

    print(f"[2/4] Fine-tuning {args.base} for {args.epochs} epochs (imgsz={args.imgsz}, "
          f"fraction={args.fraction}) on {device} ...")
    art = FT.finetune(base=args.base, epochs=args.epochs, batch=args.batch,
                      imgsz=args.imgsz, device=device, fraction=args.fraction,
                      freeze=args.freeze, lr0=args.lr0, data_yaml=data_yaml)

    print(f"[3/4] Fine-tuned model: hand-rolled eval on same test subset (native 12-class) ...")
    ft_model = Det.load_detector(art["best"])
    ft_sub = handrolled_subset_map(ft_model, art["best"], args.eval_device, coco_space=False)
    print(f"      fine-tuned mAP@0.5 = {ft_sub['mAP@0.5']:.3f}  mAP@.5:.95 = {ft_sub['mAP@[.5:.95]']:.3f}")

    print(f"[4/4] Cross-check: ultralytics val() on full held-out test split ...")
    ft_full = FT.evaluate_on_test(art["best"], device=args.eval_device, split="test", imgsz=args.imgsz)
    print(f"      (ultralytics) mAP@0.5 = {ft_full['mAP@0.5']:.3f}  mAP@.5:.95 = {ft_full['mAP@[.5:.95]']:.3f}")

    entry = {"base": args.base, "epochs": args.epochs, "imgsz": args.imgsz,
             "fraction": args.fraction, "weights": art["best"],
             "zero_shot": zs, "finetuned": ft_sub, "finetuned_full_val": ft_full}
    record["models"] = [m for m in record["models"] if m["base"] != args.base] + [entry]
    FT.save_json(record, out_path)
    print(f"\nWrote {out_path}")
    gain = ft_sub["mAP@0.5"] - zs["mAP@0.5"]
    print(f"\n=== {args.base}: zero-shot {zs['mAP@0.5']:.3f} -> fine-tuned "
          f"{ft_sub['mAP@0.5']:.3f}  (gain {gain:+.3f} mAP@0.5, same evaluator/subset) ===")


if __name__ == "__main__":
    main()
