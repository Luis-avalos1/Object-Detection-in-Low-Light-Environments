#!/usr/bin/env python3
"""One-command, cross-platform runner for the low-light detection study.

Runs the whole pipeline — or any subset of stages — on **any machine** (macOS,
Windows, Linux; CPU, Apple MPS, or NVIDIA CUDA) with no shell scripts, no
hardcoded interpreter paths, and automatic device selection. It reuses the
project's own Python modules (it does not shell out), so the same code path runs
everywhere.

    # Everything, auto-detecting the best device, sensible defaults:
    python run.py

    # Fast smoke test (5 images/class, fast enhancers, 4 detectors):
    python run.py --quick

    # Pick stages and knobs explicitly:
    python run.py --stages matrix report --detectors yolov8n yolo11n rtdetr-l \
        --enhancers original clahe_lab adaptive_gamma --per-class 25 --device auto

    # Include the (GPU-recommended) fine-tuning arm:
    python run.py --full --finetune

Stages (run in this order when selected):
    env       print platform / device / dependency diagnostics
    split     build the deterministic stratified split + YOLO export (if missing)
    matrix    detector x enhancement benchmark -> grid.json + comparison.md + heatmap
    finetune  fine-tune the base detector on ExDark (opt-in; GPU recommended)
    report    figures + CSV tables + report.md / report.html
    qualitative   GT-vs-prediction panel image

Everything is deterministic (seed 42) and every artifact lands under results/.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# --- make `import lowlight` work from the repo root on every OS -------------- #
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# Let unsupported MPS ops fall back to CPU instead of crashing (Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Default detector roster spanning architecture *families* so the comparison is
# not just "n vs s of the same net": YOLOv5(u), YOLOv8, YOLO11 (CNN one-stage,
# three generations) + RT-DETR (transformer/DETR). All auto-download via
# ultralytics on first use.
DEFAULT_DETECTORS = ["yolov8n", "yolov8s", "yolo11n", "yolov5nu"]
# --full: one detector from each architecture family/generation (2020 -> 2024)
# plus a transformer (RT-DETR), so "different model types" is a real comparison.
FAMILY_DETECTORS = ["yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n", "rtdetr-l"]

# Fast enhancer set excludes msr/msrcr — multi-scale Retinex is ~50x slower than
# everything else (≈0 FPS in the committed results) and dominates wall-clock.
FAST_ENHANCERS = ["original", "gamma", "adaptive_gamma", "brightness_contrast",
                  "hist_eq", "clahe_hsv", "clahe_lab", "gray_world", "dcp"]


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def resolve_device(requested: str) -> str:
    from lowlight import config as C
    if requested in (None, "auto"):
        return C.pick_device()
    return requested


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_env(device: str) -> None:
    hr("Environment")
    print(f"  platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  python   : {sys.version.split()[0]} @ {sys.executable}")
    try:
        import torch
        print(f"  torch    : {torch.__version__}  "
              f"cuda={torch.cuda.is_available()}  "
              f"mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"  torch    : NOT IMPORTABLE ({e})  ->  pip install -r requirements.txt")
    for mod in ("ultralytics", "cv2", "numpy", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod:9s}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"  {mod:9s}: MISSING ({e})")
    print(f"  device   : {device}  (selected)")
    if device == "mps":
        print("  note     : training on MPS is buggy in this torch/ultralytics pin; "
              "the finetune stage will train on CPU and only infer on MPS.")


def stage_split(per_class: int) -> bool:
    from lowlight import config as C
    from lowlight import dataset as D
    hr("Split + YOLO export")
    split_json = C.DATA_DIR / "splits" / "exdark_split.json"
    if not C.IMAGES_DIR.exists():
        print(f"  ERROR: dataset not found at {C.IMAGES_DIR}")
        print("  Place ExDark under data/ExDark_Dataset/<Class>/ and ground truths")
        print("  under data/ground_truths/<Class>/ (see README). Skipping.")
        return False
    if split_json.exists():
        print(f"  reusing existing split: {split_json}")
        return True
    samples = D.build_index()
    print(f"  indexed {len(samples)} image/annotation pairs")
    split = D.make_split(samples)
    (C.DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)
    D.save_split(split, split_json)
    for name, st in D.split_stats(split).items():
        print(f"    {name}: {st['images']} images")
    try:
        yaml = D.export_yolo(split)
        print(f"  exported YOLO dataset -> {yaml}")
    except Exception as e:
        print(f"  (YOLO export skipped: {e})")
    return True


def stage_matrix(detectors, enhancers, per_class, device, imgsz, primary, n_boot, progress) -> None:
    from lowlight import config as C
    from lowlight import dataset as D
    from lowlight import benchmark as B
    from lowlight import compare
    hr("Detector × Enhancement benchmark")
    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
    samples = D.stratified_subset(split["test"], per_class=per_class)
    primary = primary if primary in detectors else detectors[0]
    print(f"  {len(samples)} test images ({per_class}/class) · device={device} · "
          f"imgsz={imgsz} · bootstrap={n_boot}\n  detectors: {detectors}\n"
          f"  enhancers: {enhancers}\n  primary (full report detector): {primary}")

    matrix, cells = compare.run_matrix(detectors, enhancers, samples, device=device,
                                       imgsz=imgsz, primary=primary, n_boot=n_boot,
                                       progress=progress)
    meta = {"device": device, "subset_per_class": per_class, "imgsz": imgsz,
            "enh_detector": primary, "detectors": detectors}
    # Feed the existing report (grid.json) AND write the cross-model comparison.
    grid_path = B.save_grid([c for c in cells if c["axis"] in ("detector", "enhancement")],
                            device=device, extra_meta=meta)
    arts = compare.write_matrix_artifacts(matrix, meta=meta)
    print(f"\n  wrote {grid_path}")
    for k, v in arts.items():
        if v:
            print(f"  wrote {v}")


def _finetune_inline(base, epochs, imgsz, device, per_class_train, freeze, fraction):
    """Self-contained fine-tune that mirrors scripts/03_finetune.py."""
    import json
    from lowlight import config as C
    from lowlight import dataset as D
    from lowlight import detectors as Det
    from lowlight import finetune as FT
    from lowlight import runner as R
    hr("Fine-tuning")

    # Training on MPS is broken in this torch/ultralytics pin; train on CPU.
    train_device = "cpu" if device == "mps" else device
    eval_device = device
    if device == "mps":
        print("  MPS detected -> training on CPU (inference/eval still use MPS). "
              "For a real fine-tune use a CUDA GPU.")
    if train_device == "cpu":
        print("  WARNING: CPU fine-tuning is slow and, on a small subset, causes "
              "catastrophic forgetting (rare classes collapse to ~0 AP). Prefer a "
              "CUDA box with --ft-fraction 1.0, or pass --ft-freeze 10 --ft-per-class 120.")

    def handrolled(model, name, coco_space):
        split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
        sub = D.stratified_subset(split["test"], per_class=40)
        res = R.run_cell(model, name, "original", sub, device=eval_device,
                         coco_space=coco_space, keep_per_image=False)
        return {"mAP@0.5": res.metrics["mAP@0.5"], "mAP@[.5:.95]": res.metrics["mAP@[.5:.95]"],
                "recall@0.25": res.metrics["recall@0.25"],
                "per_class": {k: v["AP50"] for k, v in res.metrics["per_class"].items()},
                "n_images": res.metrics["n_images"]}

    out_path = C.METRICS_DIR / "finetune.json"
    record = json.load(open(out_path)) if out_path.exists() else {"models": []}

    # Build a class-balanced subset yaml if requested (avoids fraction starvation).
    data_yaml = None
    if per_class_train:
        import importlib.util
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("ft03", ROOT / "scripts" / "03_finetune.py")
        ft03 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ft03)
        data_yaml = ft03.build_balanced_yaml(per_class_train)

    print(f"[1/4] Zero-shot baseline for {base} ...")
    zs = handrolled(Det.load_detector(base.replace(".pt", "")), base, coco_space=True)
    print(f"      zero-shot mAP@0.5 = {zs['mAP@0.5']:.3f}")

    print(f"[2/4] Fine-tuning {base}: {epochs} epochs, imgsz={imgsz}, "
          f"fraction={fraction}, device={train_device} ...")
    art = FT.finetune(base=base, epochs=epochs, imgsz=imgsz, device=train_device,
                      fraction=fraction, freeze=freeze, data_yaml=data_yaml)

    print("[3/4] Fine-tuned eval (same evaluator/subset, native 12-class) ...")
    ft_sub = handrolled(Det.load_detector(art["best"]), art["best"], coco_space=False)
    print(f"      fine-tuned mAP@0.5 = {ft_sub['mAP@0.5']:.3f}")

    print("[4/4] Cross-check: ultralytics val() on full held-out test split ...")
    ft_full = FT.evaluate_on_test(art["best"], device=eval_device, split="test", imgsz=imgsz)

    entry = {"base": base, "epochs": epochs, "imgsz": imgsz, "fraction": fraction,
             "weights": art["best"], "zero_shot": zs, "finetuned": ft_sub,
             "finetuned_full_val": ft_full}
    record["models"] = [m for m in record["models"] if m["base"] != base] + [entry]
    FT.save_json(record, out_path)
    gain = ft_sub["mAP@0.5"] - zs["mAP@0.5"]
    print(f"\n=== {base}: zero-shot {zs['mAP@0.5']:.3f} -> fine-tuned "
          f"{ft_sub['mAP@0.5']:.3f}  (gain {gain:+.3f} mAP@0.5) ===")
    print(f"Wrote {out_path}")


def archive_run(tag: str, device: str, per_class: int) -> None:
    """Copy this run's artifacts into results/snapshots/<tag>/ for clean provenance.

    The top-level results/ stays the canonical (recommended: GPU/PC) run, while a
    tagged copy preserves a per-machine snapshot — useful for showing that mAP is
    identical across devices and only throughput (FPS) differs. (We use
    'snapshots' not 'runs' because .gitignore excludes any dir named runs/.)
    """
    import shutil
    from lowlight import config as C
    dst = C.RESULTS_DIR / "snapshots" / tag
    (dst / "metrics").mkdir(parents=True, exist_ok=True)
    (dst / "figures").mkdir(parents=True, exist_ok=True)
    hr(f"Archiving run -> {dst}")
    copied = 0
    for rel in ("comparison.md", "report.md", "report.html"):
        src = C.RESULTS_DIR / rel
        if src.exists():
            shutil.copy2(src, dst / rel); copied += 1
    for sub in ("metrics", "figures"):
        for src in (C.RESULTS_DIR / sub).glob("*"):
            if src.is_file():
                shutil.copy2(src, dst / sub / src.name); copied += 1
    # A small manifest so the tagged run is self-describing.
    (dst / "RUN_INFO.txt").write_text(
        f"tag: {tag}\nplatform: {platform.system()} {platform.machine()}\n"
        f"device: {device}\nimages_per_class: {per_class}\n"
        f"python: {sys.version.split()[0]}\n")
    print(f"  copied {copied} artifacts + RUN_INFO.txt")


def stage_domain(base, enhancers, epochs, imgsz, device, fraction, freeze, cross_eval) -> None:
    """The decisive A/B: fine-tune on original vs enhanced images, then compare."""
    from lowlight import domain as DOM
    hr("Domain fine-tune A/B (original-trained vs enhanced-trained)")
    train_device = "cpu" if device == "mps" else device
    if device == "mps":
        print("  MPS detected -> training on CPU. This experiment is meant for a "
              "CUDA GPU; on CPU it will be very slow. Prefer the Windows GPU box.")
    if train_device == "cpu":
        print("  WARNING: full-split training on CPU is impractical. Use --device "
              "auto on a CUDA machine, or lower --ft-fraction for a quick check.")
    print(f"  base={base} · domains={enhancers} · epochs={epochs} · imgsz={imgsz} · "
          f"fraction={fraction} · device={train_device}")
    DOM.run_domain_experiment(base=base, enhancers=tuple(enhancers), epochs=epochs,
                              imgsz=imgsz, device=train_device, fraction=fraction,
                              freeze=freeze, cross_eval=cross_eval)


def stage_report() -> None:
    from lowlight import report as Rep
    hr("Report")
    out = Rep.render_report()
    print("  markdown:", out["markdown"])
    print("  html    :", out["html"])
    print(f"  figures : {len(out['figures'])}")


def stage_qualitative(detector, device) -> None:
    from lowlight import config as C
    hr("Qualitative panel")
    import importlib.util
    spec = importlib.util.spec_from_file_location("q05", ROOT / "scripts" / "05_qualitative.py")
    sys.argv = ["05_qualitative.py", "--detector", detector, "--device", device]
    try:
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "scripts"))
        spec.loader.exec_module(mod)
        mod.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"  (qualitative skipped: {e})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ALL = ["env", "split", "matrix", "finetune", "domain", "report", "qualitative"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=None,
                    help=f"subset of {ALL} (default: all except finetune)")
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--enhancers", nargs="*", default=None)
    ap.add_argument("--primary", default=None, help="detector used for the full report (default: first)")
    ap.add_argument("--per-class", type=int, default=20, help="test images per class for the benchmark")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda | 0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bootstrap", type=int, default=100, help="bootstrap CI resamples (0 = off)")
    ap.add_argument("--no-progress", action="store_true", help="disable tqdm bars (cleaner logs)")
    ap.add_argument("--tag", default=None,
                    help="also archive this run's artifacts to results/runs/<tag>/ "
                         "(e.g. mac-mps, pc-cuda) for per-machine provenance")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (5/class, fast enhancers)")
    ap.add_argument("--full", action="store_true", help="all 12 enhancers incl. slow MSR/MSRCR + RT-DETR")
    ap.add_argument("--finetune", action="store_true", help="also run the fine-tuning stage")
    ap.add_argument("--domain", action="store_true",
                    help="run the domain A/B: fine-tune on original vs enhanced images "
                         "and compare (the decisive test; GPU recommended)")
    # finetune knobs (shared by finetune + domain stages)
    ap.add_argument("--ft-base", default="yolov8n.pt")
    ap.add_argument("--ft-epochs", type=int, default=100)
    ap.add_argument("--ft-fraction", type=float, default=1.0)
    ap.add_argument("--ft-per-class", type=int, default=None, help="balanced train subset/class")
    ap.add_argument("--ft-freeze", type=int, default=None, help="freeze first N layers (10=backbone)")
    # domain A/B knobs
    ap.add_argument("--domain-enhancers", nargs="*", default=["original", "clahe_lab"],
                    help="the two (or more) domains to train+compare (default: original clahe_lab)")
    ap.add_argument("--no-cross-eval", action="store_true",
                    help="only evaluate each model on its matched domain (skip the off-diagonal)")
    args = ap.parse_args()

    device = resolve_device(args.device)

    # Presets
    detectors = args.detectors or (FAMILY_DETECTORS if args.full else DEFAULT_DETECTORS)
    if args.enhancers:
        enhancers = args.enhancers
    elif args.full:
        from lowlight import enhancement as E
        enhancers = list(E.ENHANCERS.keys())
    else:
        enhancers = FAST_ENHANCERS
    per_class = 5 if args.quick else args.per_class
    imgsz = args.imgsz
    n_boot = args.bootstrap
    if args.quick:
        detectors = detectors[:3]
        enhancers = ["original", "clahe_lab", "adaptive_gamma"]
        imgsz = 416            # smaller -> much faster smoke test
        n_boot = 0             # skip bootstrap CIs in quick mode

    stages = args.stages
    if not stages:
        stages = [s for s in ALL if s != "finetune"]
    if args.finetune and "finetune" not in stages:
        # insert before report so the report can pick up finetune.json
        idx = stages.index("report") if "report" in stages else len(stages)
        stages.insert(idx, "finetune")
    if args.domain and "domain" not in stages:
        idx = stages.index("report") if "report" in stages else len(stages)
        stages.insert(idx, "domain")

    t0 = time.time()
    if "env" in stages:
        stage_env(device)
    if "split" in stages:
        if not stage_split(per_class):
            print("\nDataset missing — cannot continue with model stages.")
            return 1
    if "matrix" in stages:
        stage_matrix(detectors, enhancers, per_class, device, imgsz,
                     args.primary or detectors[0], n_boot, not args.no_progress)
    if "finetune" in stages:
        _finetune_inline(args.ft_base, args.ft_epochs, args.imgsz, device,
                         args.ft_per_class, args.ft_freeze, args.ft_fraction)
    if "domain" in stages:
        stage_domain(args.ft_base, args.domain_enhancers, args.ft_epochs, args.imgsz,
                     device, args.ft_fraction, args.ft_freeze, not args.no_cross_eval)
    if "report" in stages:
        stage_report()
    if "qualitative" in stages:
        stage_qualitative((args.primary or detectors[0]), device)

    if args.tag:
        archive_run(args.tag, device, per_class)

    hr(f"Done in {time.time() - t0:.0f}s")
    print("  Comparison : results/comparison.md  (+ figures/model_enhancement_matrix.png)")
    if "domain" in stages:
        print("  Domain A/B : results/domain.md  (+ metrics/domain_finetune.json)")
    print("  Full report: results/report.md / results/report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
