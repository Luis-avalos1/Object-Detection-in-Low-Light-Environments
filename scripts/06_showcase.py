#!/usr/bin/env python3
"""Render an animated showcase video of the whole pipeline.

The video walks through the three things the project actually does, using the
real model and the real evaluator:

  1. enhancing low-light images   (a before/after wipe),
  2. the detector drawing boxes   (a sped-up montage of live predictions),
  3. printing the pooled metrics  (a faux terminal + a result figure).

Detections and the COCO-style metrics shown at the end are computed *live* on
the exact images in the montage, so the number on screen is the number the
pipeline produces. Frames are streamed straight to ffmpeg (H.264 mp4); an
optimised gif for the README is derived from it.

Usage:
    python scripts/06_showcase.py                      # full 1280x720 render
    python scripts/06_showcase.py --quick              # fast low-res preview
    python scripts/06_showcase.py --weights runs/.../best.pt   # fine-tuned model
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import cv2
import numpy as np

from lowlight import config as C
from lowlight import dataset as D
from lowlight import detectors as Det
from lowlight import enhancement as E
from lowlight import showcase as S
from lowlight.metrics import DetectionEvaluator, ImageEval, coco_dets_to_exdark

# Enhancement methods cycled through in stage 1 (each is visually distinct and
# produces a natural-looking result a detector can use). Ordered to lead with the
# cleanest-looking results and still feature the project's MSRCR.
ENHANCE_METHODS = ["clahe_lab", "adaptive_gamma", "msrcr", "gray_world"]


def find_ffmpeg(explicit: str | None) -> str:
    for cand in (explicit, "ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe"):
        if cand and (shutil.which(cand) or Path(cand).exists()):
            return cand
    raise SystemExit("ffmpeg not found — install it or pass --ffmpeg <path>")


def round_robin(by_class: dict[str, list]) -> list:
    """Interleave class lists so consecutive picks are different classes."""
    out, lists = [], [list(v) for v in by_class.values()]
    i = 0
    while any(lists):
        lst = lists[i % len(lists)]
        if lst:
            out.append(lst.pop(0))
        i += 1
        if i > 10000:
            break
    return out


def imread(path: str, max_side: int = 1280) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:                 # keep work cheap; fit() downsamples anyway
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


class FFmpegWriter:
    """Stream raw BGR frames to ffmpeg, encoding H.264 (yuv420p) mp4."""

    def __init__(self, path: Path, w: int, h: int, fps: int, ffmpeg: str):
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-preset", "medium", "-movflags", "+faststart", str(path),
        ]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg encode failed:\n" + err.decode(errors="ignore"))


def make_gif(ffmpeg: str, mp4: Path, gif: Path, speed: float, fps: int, width: int):
    """Derive an optimised, palette-quantised gif from the mp4."""
    vf = (
        f"setpts={speed}*PTS,fps={fps},scale={width}:-2:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)]
    subprocess.run(cmd, check=True, capture_output=True)


# --------------------------------------------------------------------------- #
def collect_detections(model, samples, device, want, log):
    """Run the detector on candidates, keep diverse non-empty ones, and build
    DetItems + a populated evaluator. Returns (items, evaluator)."""
    coco_space = Det.is_coco_model(model)
    names = dict(C.EXDARK_ID_TO_CLASS)
    ev = DetectionEvaluator()
    by_class: dict[str, list[S.DetItem]] = {}

    for s in samples:
        img = imread(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        t0 = time.time()
        det = Det.predict(model, img, device=device)
        infer_ms = (time.time() - t0) * 1000
        if coco_space:
            b, sc, lb = coco_dets_to_exdark(det.boxes, det.scores, det.labels)
        else:
            b, sc, lb = det.boxes, det.scores, det.labels
        gt = s.load_objects(clip_to=(w, h))
        gtb = np.array([[o.x1, o.y1, o.x2, o.y2] for o in gt], np.float32).reshape(-1, 4)
        gtl = np.array([o.exdark_id for o in gt], int)
        item = S.DetItem(image=img, boxes=b, scores=sc, labels=lb,
                         gt_boxes=gtb, gt_labels=gtl, names=names,
                         klass=s.folder_class, infer_ms=infer_ms)
        n_conf = int((sc >= 0.25).sum())   # confident dets -> what the montage will draw
        by_class.setdefault(s.folder_class, []).append((item, n_conf, gtb, gtl, b, sc, lb))

    # prefer images where the model actually found something, keep class variety
    for v in by_class.values():
        v.sort(key=lambda r: -r[1])
    picked = round_robin({k: v for k, v in by_class.items()})

    items: list[S.DetItem] = []
    for item, n_conf, gtb, gtl, b, sc, lb in picked:
        if len(items) >= want:
            break
        if n_conf == 0 and len(items) >= want - 2:   # allow a couple of empties only if short
            continue
        items.append(item)
        ev.add(ImageEval(b.reshape(-1, 4), sc.reshape(-1), lb.reshape(-1), gtb, gtl))
        log(f"    + {item.klass:<9} {n_conf:2d} det>=.25  {item.infer_ms:5.0f} ms  ({item.image.shape[1]}x{item.image.shape[0]})")
    return items, ev


def collect_enhancements(samples, want, log):
    """Pick clearly-dark (but not crushed-to-noise) images, one per class, and
    apply a rotating set of enhancers."""
    scored = []
    for s in samples:
        img = imread(s.image_path)
        if img is None:
            continue
        scored.append((S._luma(img), img, s.folder_class))
    # sweet spot: dark enough to read as low-light, bright enough that enhancing
    # recovers detail instead of amplifying sensor noise.
    band = sorted((r for r in scored if 18 <= r[0] <= 85), key=lambda r: r[0])
    pool = band if len(band) >= want else sorted(scored, key=lambda r: r[0])
    items: list[S.EnhanceItem] = []
    seen = set()
    for luma, img, klass in pool:
        if klass in seen:                          # one image per class for variety
            continue
        method = ENHANCE_METHODS[len(items) % len(ENHANCE_METHODS)]
        enh = E.get(method)(img)
        items.append(S.EnhanceItem(original=img, enhanced=enh, method=method, klass=klass))
        seen.add(klass)
        log(f"    + {klass:<9} luma {luma:5.1f}  ->  {method}")
        if len(items) >= want:
            break
    return items


def load_figure() -> np.ndarray | None:
    for name in ("per_class_ap_heatmap.png", "map_by_enhancement.png", "accuracy_vs_latency.png"):
        p = C.FIGURES_DIR / name
        if p.exists():
            return cv2.imread(str(p))
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="yolov8n")
    ap.add_argument("--weights", default=None, help="fine-tuned best.pt (native ExDark labels)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-detect", type=int, default=14, help="images in the detection montage")
    ap.add_argument("--n-enhance", type=int, default=4, help="images in the enhancement stage")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "showcase" / "showcase.mp4"))
    ap.add_argument("--quick", action="store_true", help="fast low-res preview")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()

    if args.quick:
        args.width, args.height, args.fps = 854, 480, 20
        args.n_detect, args.n_enhance = 4, 2

    cfg = S.Cfg(w=args.width - args.width % 2, h=args.height - args.height % 2, fps=args.fps)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def log(msg=""):
        print(msg, flush=True)

    log(f"[1/5] loading detector: {args.weights or args.detector}")
    model = Det.load_detector(args.weights or args.detector)

    log("[2/5] selecting samples + running inference (this is the real model)")
    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")
    det_pool = D.stratified_subset(split["test"], per_class=1 if args.quick else 3)
    enh_pool = D.stratified_subset(split["test"], per_class=1 if args.quick else 2)
    det_items, ev = collect_detections(model, det_pool, args.device, args.n_detect, log)
    enh_items = collect_enhancements(enh_pool, args.n_enhance, log)
    metrics = ev.compute()
    log(f"      metrics: mAP@0.5={metrics['mAP@0.5']:.3f}  "
        f"mAP@[.5:.95]={metrics['mAP@[.5:.95]']:.3f}  "
        f"P={metrics['precision@0.25']:.3f} R={metrics['recall@0.25']:.3f}  "
        f"({metrics['n_images']} imgs, {metrics['n_gt_total']} GT)")

    figure = load_figure()
    detector_name = "fine-tuned" if args.weights else args.detector

    log("[3/5] rendering frames -> ffmpeg (H.264)")
    writer = FFmpegWriter(out, cfg.w, cfg.h, cfg.fps, ffmpeg)
    t_render = time.time()
    clock = 0.0

    def run(gen):
        nonlocal clock
        start = writer.n
        for frame in gen:
            writer.write(frame)
            if writer.n % 60 == 0:
                print(f"      {writer.n} frames  ({clock:5.1f}s)", flush=True)
        clock = writer.n / cfg.fps
        return writer.n - start

    run(S.scene_title(cfg, clock0=clock))
    run(S.scene_enhancement(cfg, enh_items, clock0=clock))
    run(S.scene_detection(cfg, det_items, clock0=clock, top_k=12))
    run(S.scene_results(cfg, metrics, detector_name, figure, clock0=clock))
    run(S.scene_outro(cfg, metrics, clock0=clock))
    writer.close()

    dur = writer.n / cfg.fps
    size_mb = out.stat().st_size / 1e6
    log(f"      wrote {out}  ({writer.n} frames, {dur:.1f}s, {size_mb:.1f} MB, "
        f"{time.time() - t_render:.0f}s render)")

    if not args.no_gif:
        gif = out.with_suffix(".gif")
        log(f"[4/5] deriving gif -> {gif}")
        # speed the gif up a touch and cap fps/width so the README stays light
        make_gif(ffmpeg, out, gif, speed=0.62, fps=12, width=args.gif_width)
        log(f"      wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")

    log("[5/5] done.")
    log("\nshareable artifacts:")
    log(f"  video : {out}")
    if not args.no_gif:
        log(f"  gif   : {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
