"""Animated showcase video for the low-light detection pipeline.

This module renders a self-contained "product demo" of the project as a sequence
of frames (BGR ``uint8`` numpy arrays of a fixed canvas size) that a caller can
stream to ``ffmpeg``. It deliberately depends only on OpenCV + numpy so it runs
anywhere the rest of the project does — no fonts, no extra packages.

The video is built from five scenes, each a generator that ``yield``s frames:

1. :func:`scene_title`        — animated title card.
2. :func:`scene_enhancement`  — a wipe that reveals each enhanced image growing
                                out of its dark original (the "enhancement
                                happening" section).
3. :func:`scene_detection`    — the centrepiece: for every test image a scan
                                line sweeps, then the detector's boxes are drawn
                                on one at a time with their labels — a sped-up
                                montage of the model predicting.
4. :func:`scene_results`      — a faux terminal types out the run and prints the
                                real, pooled COCO-style metrics, then a result
                                figure slides in.
5. :func:`scene_outro`        — links / call-to-action card.

All animation is time-parameterised through :class:`Cfg`, so the same code
produces a quick low-res preview or a full HD render.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Palette (BGR) — a dark, GitHub-ish UI with an orange "prediction" accent.
# --------------------------------------------------------------------------- #
BG = (20, 16, 13)          # near-black background
PANEL = (34, 28, 23)       # raised panel
PANEL_HI = (52, 43, 35)    # lighter panel / hover
LINE = (64, 54, 46)        # hairline borders
FG = (236, 236, 236)       # primary text
MUTED = (158, 152, 148)    # secondary text
FAINT = (96, 92, 89)       # tertiary / disabled
ACCENT = (0, 140, 255)     # orange  — predictions / highlights
GREEN = (90, 205, 105)     # green   — ground truth / success
CYAN = (224, 196, 64)      # teal    — numbers / accents
TERM_BG = (24, 18, 14)     # terminal panel background

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

GH_PAGES = "luis-avalos1.github.io/Object-Detection-in-Low-Light-Environments"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    """Canvas + timing configuration. Durations are in seconds."""
    w: int = 1280
    h: int = 720
    fps: int = 30

    def n(self, seconds: float) -> int:
        """Seconds -> a whole number of frames (>= 1)."""
        return max(1, int(round(seconds * self.fps)))


# --------------------------------------------------------------------------- #
# Easing + small math
# --------------------------------------------------------------------------- #
def smooth(t: float) -> float:
    """Smoothstep ease in/out, clamped to [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def canvas(cfg: Cfg) -> np.ndarray:
    img = np.empty((cfg.h, cfg.w, 3), np.uint8)
    img[:] = BG
    return img


def text_size(s: str, scale: float, thick: int, font=FONT) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    return w, h


def text(img, s, org, scale=0.6, color=FG, thick=1, font=FONT, shadow=True):
    """Anti-aliased text with an optional drop shadow for legibility."""
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, AA)
    return text_size(s, scale, thick, font)


def text_center(img, s, cx, y, scale=0.6, color=FG, thick=1, font=FONT, shadow=True):
    w, _ = text_size(s, scale, thick, font)
    return text(img, s, (int(cx - w / 2), y), scale, color, thick, font, shadow)


def rrect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Filled or outlined rounded rectangle."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, AA)
        return
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, AA)
        for cx, cy, a0 in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                           (x1 + r, y2 - r, 90), (x2 - r, y2 - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, thickness, AA)


def chip(img, s, x, y, fg=FG, bg=PANEL_HI, scale=0.5, pad=6, thick=1, font=FONT):
    """Pill-shaped label. ``(x, y)`` is the top-left. Returns (w, h)."""
    tw, th = text_size(s, scale, thick, font)
    w, h = tw + 2 * pad + 6, th + 2 * pad
    rrect(img, x, y, x + w, y + h, h / 2, bg, -1)
    text(img, s, (x + pad + 3, y + h - pad - 1), scale, fg, thick, font, shadow=False)
    return w, h


def fit(src: np.ndarray, box_w: int, box_h: int) -> tuple[np.ndarray, float]:
    """Resize ``src`` to fit inside ``box_w x box_h`` preserving aspect ratio."""
    h, w = src.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(src, (max(1, int(w * s)), max(1, int(h * s)))), s


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    """Paste ``src`` onto ``dst`` at (x, y) with clipping at the borders."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def fade(img: np.ndarray, k: float) -> np.ndarray:
    """Fade a frame toward black. ``k`` is the visible fraction (1 = full)."""
    k = min(max(k, 0.0), 1.0)
    if k >= 1.0:
        return img
    return (img.astype(np.float32) * k).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Shared chrome: header bar + footer
# --------------------------------------------------------------------------- #
def chrome(cfg: Cfg, kicker: str, title: str, step: str, t_clock: float) -> np.ndarray:
    """Fresh canvas with the persistent header + footer drawn."""
    img = canvas(cfg)
    # header
    text(img, kicker, (40, 52), 0.5, ACCENT, 1, FONT_BOLD)
    text(img, title, (40, 84), 0.82, FG, 1, FONT_BOLD)
    if step:
        sw, _ = text_size(step, 0.5, 1, FONT_BOLD)
        text(img, step, (cfg.w - 40 - sw, 84), 0.5, MUTED, 1, FONT_BOLD)
    cv2.line(img, (40, 100), (cfg.w - 40, 100), LINE, 1, AA)
    # footer
    cv2.line(img, (40, cfg.h - 42), (cfg.w - 42, cfg.h - 42), LINE, 1, AA)
    text(img, "Object-Detection-in-Low-Light-Environments", (40, cfg.h - 18), 0.44, FAINT, 1)
    clock = f"{int(t_clock) // 60:02d}:{int(t_clock) % 60:02d}"
    cw, _ = text_size(clock, 0.44, 1)
    cv2.circle(img, (cfg.w - 56 - cw, cfg.h - 22), 4, ACCENT, -1, AA)
    text(img, clock, (cfg.w - 44 - cw, cfg.h - 18), 0.44, MUTED, 1)
    return img


def content_box(cfg: Cfg) -> tuple[int, int, int, int]:
    """The (x, y, w, h) region between header and footer for scene content."""
    return 40, 116, cfg.w - 80, cfg.h - 116 - 54


# --------------------------------------------------------------------------- #
# Scene 1 — title card
# --------------------------------------------------------------------------- #
def scene_title(cfg: Cfg, seconds: float = 3.4, clock0: float = 0.0) -> Iterator[np.ndarray]:
    n = cfg.n(seconds)
    cx = cfg.w // 2
    for i in range(n):
        t = i / (n - 1)
        img = canvas(cfg)
        # animated horizon glow
        glow = int(40 + 24 * smooth(min(1.0, t * 2)))
        cv2.line(img, (0, cfg.h // 2 + 70), (cfg.w, cfg.h // 2 + 70), (glow, glow // 2, glow // 4), 2, AA)

        a_title = smooth((t - 0.05) / 0.45)
        a_sub = smooth((t - 0.35) / 0.45)
        a_tag = smooth((t - 0.55) / 0.4)
        text_center(img, "OBJECT DETECTION IN", cx, cfg.h // 2 - 48,
                    1.15, fade_color(FG, a_title), 2, FONT_BOLD)
        text_center(img, "LOW-LIGHT ENVIRONMENTS", cx, cfg.h // 2 + 6,
                    1.15, fade_color(ACCENT, a_title), 2, FONT_BOLD)
        if a_sub > 0:
            text_center(img, "image enhancement  x  detector benchmarking  on  ExDark",
                        cx, cfg.h // 2 + 56, 0.6, fade_color(MUTED, a_sub), 1)
        if a_tag > 0:
            chip_text = "a corrected COCO-style mAP study"
            cw, _ = text_size(chip_text, 0.5, 1)
            chip(img, chip_text, cx - cw // 2 - 9, cfg.h // 2 + 84,
                 fg=fade_color(BG, a_tag), bg=fade_color(CYAN, a_tag), scale=0.5)

        k = min(smooth(t / 0.18), smooth((1 - t) / 0.12))
        yield fade(img, max(k, 0.0))


def fade_color(color, a: float):
    a = min(max(a, 0.0), 1.0)
    return tuple(int(c * a) for c in color)


# --------------------------------------------------------------------------- #
# Scene 2 — enhancement reveal (wipe)
# --------------------------------------------------------------------------- #
@dataclass
class EnhanceItem:
    original: np.ndarray   # BGR uint8
    enhanced: np.ndarray   # BGR uint8 (same HxW)
    method: str
    klass: str


def _luma(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def scene_enhancement(cfg: Cfg, items: Sequence[EnhanceItem], clock0: float = 0.0) -> Iterator[np.ndarray]:
    bx, by, bw, bh = content_box(cfg)
    fps = cfg.fps
    per = cfg.n(3.0)               # frames per item
    sweep = cfg.n(1.5)
    fin, fout = cfg.n(0.22), cfg.n(0.18)
    total_clock = clock0
    for it in items:
        disp_o, _ = fit(it.original, bw, bh - 40)
        disp_e, _ = fit(it.enhanced, bw, bh - 40)
        ih, iw = disp_o.shape[:2]
        px = bx + (bw - iw) // 2
        py = by + (bh - 40 - ih) // 2 + 30
        lo, le = _luma(it.original), _luma(it.enhanced)
        for i in range(per):
            t = i / (per - 1)
            frac = ease_out(min(1.0, i / sweep))
            img = chrome(cfg, "STAGE 01", "Low-light image enhancement", "enhance",
                         total_clock + i / fps)
            text(img, f"{it.klass}   |   method: {it.method}", (bx, by + 16), 0.6, MUTED, 1)

            combined = disp_o.copy()
            dvd = int(frac * iw)
            if dvd > 0:
                combined[:, :dvd] = disp_e[:, :dvd]
            cv2.rectangle(combined, (0, 0), (iw - 1, ih - 1), LINE, 1)
            paste(img, combined, px, py)
            # moving divider with glow
            if 0 < dvd < iw:
                lx = px + dvd
                cv2.line(img, (lx, py), (lx, py + ih), (255, 255, 255), 2, AA)
                cv2.line(img, (lx, py), (lx, py + ih), ACCENT, 1, AA)
            # labels riding the divider
            chip(img, "BEFORE", px + 8, py + 8, fg=FG, bg=(40, 40, 40), scale=0.46)
            aw, _ = text_size(it.method.upper(), 0.46, 1)
            chip(img, it.method.upper(), px + iw - aw - 24, py + 8, fg=BG, bg=ACCENT, scale=0.46)
            # luminance readout climbing with the wipe
            cur = lerp(lo, le, frac)
            text(img, f"mean luminance  {lo:5.1f}  ->  {cur:5.1f}",
                 (px + 8, py + ih - 12), 0.5, fade_color(CYAN, 0.5 + 0.5 * frac), 1)

            k = min(smooth((i + 1) / fin), smooth((per - i) / fout))
            yield fade(img, k)
        total_clock += per / fps


# --------------------------------------------------------------------------- #
# Scene 3 — detection montage (the model drawing boxes)
# --------------------------------------------------------------------------- #
@dataclass
class DetItem:
    image: np.ndarray            # BGR uint8 (original)
    boxes: np.ndarray            # (N,4) xyxy in original pixels, sorted desc score
    scores: np.ndarray           # (N,)
    labels: np.ndarray           # (N,) exdark ids
    gt_boxes: np.ndarray         # (M,4) xyxy
    gt_labels: np.ndarray        # (M,)
    names: dict                  # id -> name
    klass: str
    infer_ms: float


def _map_box(b, scale, px, py):
    return (int(px + b[0] * scale), int(py + b[1] * scale),
            int(px + b[2] * scale), int(py + b[3] * scale))


def _reveal_box(img, fb, label, color, e):
    """Draw one prediction box growing from its centre with alpha ``e``."""
    x1, y1, x2, y2 = fb
    if x2 <= x1 or y2 <= y1:
        return
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    g = smooth(e)
    rx1, ry1 = int(lerp(cx, x1, g)), int(lerp(cy, y1, g))
    rx2, ry2 = int(lerp(cx, x2, g)), int(lerp(cy, y2, g))
    # ROI big enough for the box + a label chip above it
    H, W = img.shape[:2]
    bx1, by1 = max(0, x1 - 3), max(0, y1 - 26)
    bx2, by2 = min(W, x2 + 140), min(H, y2 + 4)
    if bx2 <= bx1 or by2 <= by1:
        return
    roi = img[by1:by2, bx1:bx2]
    tmp = roi.copy()
    cv2.rectangle(tmp, (rx1 - bx1, ry1 - by1), (rx2 - bx1, ry2 - by1), color, 2, AA)
    if e > 0.55:
        cw, ch = text_size(label, 0.46, 1)
        ly = max(0, y1 - by1 - ch - 8)
        rrect(tmp, x1 - bx1, ly, x1 - bx1 + cw + 12, ly + ch + 8, 3, color, -1)
        text(tmp, label, (x1 - bx1 + 6, ly + ch + 3), 0.46, BG, 1, shadow=False)
    a = min(1.0, e * 1.15)
    img[by1:by2, bx1:bx2] = cv2.addWeighted(tmp, a, roi, 1 - a, 0)


def scene_detection(cfg: Cfg, items: Sequence[DetItem], clock0: float = 0.0,
                    top_k: int = 14) -> Iterator[np.ndarray]:
    bx, by, bw, bh = content_box(cfg)
    fps = cfg.fps
    scan_f = cfg.n(0.45)
    grow_f = cfg.n(0.16)
    step_f = max(1, cfg.n(0.07))
    hold_f = cfg.n(0.55)
    fin = cfg.n(0.12)
    total = clock0
    n_items = len(items)
    for idx, it in enumerate(items):
        disp, scale = fit(it.image, bw, bh - 24)
        ih, iw = disp.shape[:2]
        px = bx + (bw - iw) // 2
        py = by + (bh - 24 - ih) // 2 + 18

        order = np.argsort(-it.scores) if len(it.scores) else np.array([], int)
        order = order[it.scores[order] >= 0.25][:top_k]   # only confident boxes (matches HUD)
        fboxes = [_map_box(it.boxes[i], scale, px, py) for i in order]
        flabels = [f"{it.names.get(int(it.labels[i]), it.labels[i])} {it.scores[i]:.2f}" for i in order]
        gt_mapped = [_map_box(g, scale, px, py) for g in it.gt_boxes] if len(it.gt_boxes) else []

        n_boxes = len(fboxes)
        reveal_span = max(grow_f, n_boxes * step_f + grow_f)
        per = scan_f + reveal_span + hold_f

        for i in range(per):
            img = chrome(cfg, "STAGE 02", "Detector drawing predictions", "detect",
                         total + i / fps)
            # HUD: counters
            text(img, f"image {idx + 1:02d} / {n_items:02d}", (bx, by + 10), 0.54, MUTED, 1)
            rtext = f"{it.infer_ms:.0f} ms / frame   |   conf >= 0.25"
            rw, _ = text_size(rtext, 0.5, 1)
            text(img, rtext, (bx + bw - rw, by + 10), 0.5, FAINT, 1)

            scanning = i < scan_f
            base = disp if not scanning else (disp.astype(np.float32) * 0.5).astype(np.uint8)
            frame_img = base.copy()
            cv2.rectangle(frame_img, (0, 0), (iw - 1, ih - 1), LINE, 1)
            paste(img, frame_img, px, py)

            if scanning:
                sy = py + int((i / scan_f) * ih)
                cv2.line(img, (px, sy), (px + iw, sy), ACCENT, 2, AA)
                dots = "." * (1 + (i % 3))
                chip(img, f"running inference{dots}", px + 8, py + 8, fg=BG, bg=ACCENT, scale=0.48)
            else:
                # faint ground truth underneath the predictions
                for g in gt_mapped:
                    cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), GREEN, 1, AA)
                rf = i - scan_f
                shown = 0
                for bi, (fb, lab) in enumerate(zip(fboxes, flabels)):
                    e = (rf - bi * step_f) / max(1, grow_f)
                    if e <= 0:
                        continue
                    _reveal_box(img, fb, lab, ACCENT, min(1.0, e))
                    if e >= 1.0:
                        shown += 1
                # legend + live detection count
                chip(img, "prediction", px + 8, py + ih - 30, fg=BG, bg=ACCENT, scale=0.44)
                chip(img, "ground truth", px + 122, py + ih - 30, fg=BG, bg=GREEN, scale=0.44)
                cnt = f"{shown} object" + ("" if shown == 1 else "s")
                cw, _ = text_size(cnt, 0.6, 2, FONT_BOLD)
                text(img, cnt, (px + iw - cw - 10, py + ih - 14), 0.6, ACCENT, 2, FONT_BOLD)

            yield fade(img, smooth((i + 1) / fin))
        total += per / fps


# --------------------------------------------------------------------------- #
# Scene 4 — results terminal + figure
# --------------------------------------------------------------------------- #
def _metric_lines(metrics: dict, detector: str) -> list[tuple[str, tuple]]:
    """Build the (text, color) lines the terminal types out."""
    pc = metrics.get("per_class", {})
    ranked = sorted(
        ((k, v.get("AP50")) for k, v in pc.items() if v.get("AP50") is not None),
        key=lambda kv: kv[1], reverse=True,
    )
    top = "  ".join(f"{k} {v:.2f}" for k, v in ranked[:6])
    L: list[tuple[str, tuple]] = [
        (f"$ python scripts/02_run_benchmark.py --detector {detector}", FG),
        (f"  loaded {detector}  (COCO-pretrained adapter)", MUTED),
        (f"  evaluating {metrics.get('n_images', 0)} ExDark images"
         f"   |   {metrics.get('n_gt_total', 0)} GT objects", MUTED),
        ("  pooling detections -> COCO-style evaluator ...", MUTED),
        ("", FG),
        (f"  mAP@0.5 .............. {metrics.get('mAP@0.5', 0):.3f}", CYAN),
        (f"  mAP@[.5:.95] ......... {metrics.get('mAP@[.5:.95]', 0):.3f}", CYAN),
        (f"  precision@0.25 ....... {metrics.get('precision@0.25', 0):.3f}", FG),
        (f"  recall@0.25 .......... {metrics.get('recall@0.25', 0):.3f}", FG),
        (f"  F1@0.25 .............. {metrics.get('f1@0.25', 0):.3f}", FG),
        ("", FG),
        ("  per-class AP@0.5 (Cup & Boat recovered):", MUTED),
        (f"    {top}", GREEN),
        ("  done -> results/report.md", ACCENT),
    ]
    return L


def scene_results(cfg: Cfg, metrics: dict, detector: str,
                  figure: np.ndarray | None, clock0: float = 0.0,
                  seconds: float = 9.5) -> Iterator[np.ndarray]:
    lines = _metric_lines(metrics, detector)
    bx, by, bw, bh = content_box(cfg)
    # terminal panel (left ~58%) and figure panel (right)
    term_w = int(bw * 0.56)
    tx1, ty1 = bx, by + 8
    tx2, ty2 = bx + term_w, by + bh
    line_h = 30
    total_chars = sum(len(s) + 1 for s, _ in lines)

    n = cfg.n(seconds)
    type_n = int(n * 0.62)
    fps = cfg.fps

    fig_disp = None
    if figure is not None:
        fig_disp, _ = fit(figure, bw - term_w - 28, bh - 16)

    for i in range(n):
        img = chrome(cfg, "STAGE 03", "Results & metrics", "report", clock0 + i / fps)
        # terminal shell
        rrect(img, tx1, ty1, tx2, ty2, 10, TERM_BG, -1)
        rrect(img, tx1, ty1, tx2, ty2, 10, LINE, 1)
        rrect(img, tx1, ty1, tx2, ty1 + 30, 10, PANEL, -1)
        for j, c in enumerate(((80, 95, 245), (70, 180, 235), (90, 200, 110))):
            cv2.circle(img, (tx1 + 18 + j * 18, ty1 + 15), 5, c, -1, AA)
        text(img, "benchmark - bash", (tx1 + 78, ty1 + 20), 0.46, MUTED, 1, shadow=False)

        # how many characters are revealed so far
        reveal = total_chars if i >= type_n else int(total_chars * (i / type_n))
        consumed, cursor = 0, None
        y = ty1 + 56
        for s, color in lines:
            if consumed >= reveal and s:
                break
            take = min(len(s), max(0, reveal - consumed))
            if y < ty2 - 10:
                text(img, s[:take], (tx1 + 16, y), 0.5, color, 1, FONT, shadow=False)
                if take < len(s) or (consumed <= reveal < consumed + len(s) + 1):
                    cw, _ = text_size(s[:take], 0.5, 1)
                    cursor = (tx1 + 18 + cw, y)
            consumed += len(s) + 1
            y += line_h
        # blinking cursor
        if cursor and (i // max(1, fps // 3)) % 2 == 0:
            cv2.rectangle(img, (cursor[0], cursor[1] - 13),
                          (cursor[0] + 8, cursor[1] + 2), FG, -1)

        # figure slides in from the right once typing is mostly done
        if fig_disp is not None:
            slide = smooth((i - type_n * 0.7) / max(1, n - type_n * 0.7))
            if slide > 0:
                fh, fw = fig_disp.shape[:2]
                fxt = tx2 + 20
                fx = int(lerp(cfg.w + 20, fxt, slide))
                fy = by + (bh - fh) // 2
                card = img.copy()
                rrect(card, fxt - 10, fy - 10, fxt + fw + 10, fy + fh + 10, 10, PANEL, -1)
                paste(card, fig_disp, fx, fy)
                img = cv2.addWeighted(card, slide, img, 1 - slide, 0)
                if slide > 0.6:
                    text(img, "results/figures/", (fxt, by + bh - 2), 0.44,
                         fade_color(MUTED, (slide - 0.6) / 0.4), 1)

        k = min(smooth((i + 1) / cfg.n(0.2)), smooth((n - i) / cfg.n(0.2)))
        yield fade(img, k)


# --------------------------------------------------------------------------- #
# Scene 5 — outro
# --------------------------------------------------------------------------- #
def scene_outro(cfg: Cfg, metrics: dict, seconds: float = 3.8,
                clock0: float = 0.0) -> Iterator[np.ndarray]:
    n = cfg.n(seconds)
    cx = cfg.w // 2
    rows = [
        ("Live site", GH_PAGES, ACCENT),
        ("Full report", "results/report.md", CYAN),
        ("Methodology", "docs/METHODOLOGY.md", CYAN),
    ]
    for i in range(n):
        t = i / (n - 1)
        img = canvas(cfg)
        a = smooth(t / 0.3)
        text_center(img, "Fine-tuning is the lever. Enhancement on a fixed",
                    cx, cfg.h // 2 - 120, 0.66, fade_color(MUTED, a), 1)
        text_center(img, "detector is roughly neutral.",
                    cx, cfg.h // 2 - 90, 0.66, fade_color(MUTED, a), 1)
        # "mAP50 / mAP50-95" (ultralytics notation) avoids "@", which the Hershey
        # fonts render as a blob at large sizes.
        head = (f"mAP50  {metrics.get('mAP@0.5', 0):.3f}"
                f"      |      mAP50-95  {metrics.get('mAP@[.5:.95]', 0):.3f}")
        text_center(img, head, cx, cfg.h // 2 - 34, 0.82, fade_color(FG, a), 2, FONT_BOLD)
        for r, (k, v, col) in enumerate(rows):
            ar = smooth((t - 0.25 - r * 0.12) / 0.4)
            if ar <= 0:
                continue
            row = f"{k:<12}  {v}"
            text_center(img, row, cx, cfg.h // 2 + 26 + r * 34, 0.6, fade_color(col, ar), 1, FONT_BOLD)
        k = min(smooth((i + 1) / cfg.n(0.25)), smooth((n - i) / cfg.n(0.4)))
        yield fade(img, k)
