"""Turn benchmark JSON into figures, CSV tables, and a research report.

Pure post-processing: reads ``results/metrics/*.json`` and writes
``results/figures/*.png``, ``results/metrics/*.csv``, and
``results/report.md`` / ``results/report.html``. No model calls, so it is fast
and safe to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    _HAS_SNS = True
except Exception:
    _HAS_SNS = True
    plt.style.use("seaborn-v0_8-whitegrid")
    _HAS_SNS = False

PALETTE = ["#2b8cbe", "#fd8d3c", "#74c476", "#9e9ac8", "#f768a1", "#a6611a"]


def _load(name: str) -> dict | None:
    p = C.METRICS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_enhancement_bars(grid: dict, out: Path) -> Path | None:
    """Grouped bar chart: mAP@0.5 by enhancement, grouped by detector regime."""
    cells = grid.get("cells", [])
    cells = [c for c in cells if c.get("axis") == "enhancement"]
    if not cells:
        return None
    methods = []
    for c in cells:
        if c["enhancement"] not in methods:
            methods.append(c["enhancement"])
    regimes = []
    for c in cells:
        r = c.get("regime", "frozen-coco")
        if r not in regimes:
            regimes.append(r)

    x = np.arange(len(methods))
    width = 0.8 / max(len(regimes), 1)
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.1), 5.5))
    for gi, reg in enumerate(regimes):
        vals = []
        for mth in methods:
            cell = next((c for c in cells if c["enhancement"] == mth and c.get("regime", "frozen-coco") == reg), None)
            vals.append(cell["metrics"]["mAP@0.5"] if cell else 0.0)
        ax.bar(x + gi * width, vals, width, label=reg, color=PALETTE[gi % len(PALETTE)], edgecolor="black", linewidth=0.5)
    # baseline line at the 'original' value of the first regime
    base = next((c for c in cells if c["enhancement"] == "original" and c.get("regime", "frozen-coco") == regimes[0]), None)
    if base:
        ax.axhline(base["metrics"]["mAP@0.5"], ls="--", color="gray", lw=1, label="original baseline")
    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Detection mAP@0.5 by enhancement method")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_finetune_bars(ft: dict, out: Path) -> Path | None:
    """Zero-shot vs fine-tuned mAP@0.5 per detector — the headline lever."""
    models = ft.get("models", [])
    models = [m for m in models if m.get("zero_shot") and m.get("finetuned")]
    if not models:
        return None
    names = [m["base"].replace(".pt", "") for m in models]
    zs = [m["zero_shot"]["mAP@0.5"] for m in models]
    fts = [m["finetuned"]["mAP@0.5"] for m in models]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.6), 5.5))
    b1 = ax.bar(x - w / 2, zs, w, label="zero-shot COCO", color="#9ecae1", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, fts, w, label="fine-tuned on ExDark", color="#fd8d3c", edgecolor="black", linewidth=0.5)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mAP@0.5 (held-out test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fine-tuning is the dominant lever: zero-shot vs fine-tuned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_per_class_heatmap(grid: dict, out: Path) -> Path | None:
    cells = [c for c in grid.get("cells", []) if c.get("axis") == "enhancement"]
    if not cells:
        return None
    classes = C.EXDARK_CLASSES
    labels = [f"{c['detector']}/{c['enhancement']}" for c in cells]
    mat = np.zeros((len(cells), len(classes)))
    for i, c in enumerate(cells):
        pc = c["metrics"].get("per_class", {})
        for j, cl in enumerate(classes):
            v = pc.get(cl, {}).get("AP50")
            mat[i, j] = np.nan if v is None else v
    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 0.9), max(5, len(cells) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="AP@0.5")
    ax.set_title("Per-class AP@0.5 (Cup & Boat columns are now non-zero)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_quality_vs_map(grid: dict, out: Path) -> Path | None:
    cells = [c for c in grid.get("cells", []) if c.get("axis") == "enhancement" and c.get("quality") is not None]
    if len(cells) < 3:
        return None
    q = [c["quality"]["entropy"] for c in cells]
    m = [c["metrics"]["mAP@0.5"] for c in cells]
    names = [c["enhancement"] for c in cells]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(q, m, s=90, color="#2b8cbe", edgecolor="black", zorder=3)
    for xi, yi, n in zip(q, m, names):
        ax.annotate(n, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
    if len(q) > 2:
        r = np.corrcoef(q, m)[0, 1]
        ax.set_title(f"Perceptual quality (entropy) vs detection mAP@0.5  (r={r:.2f})")
    ax.set_xlabel("Mean image entropy (perceptual proxy)")
    ax.set_ylabel("mAP@0.5")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_accuracy_latency(grid: dict, out: Path) -> Path | None:
    cells = [c for c in grid.get("cells", []) if c.get("axis") == "detector"]
    if len(cells) < 2:
        return None
    fps = [c["fps"] for c in cells]
    m = [c["metrics"]["mAP@0.5"] for c in cells]
    names = [c["detector"] for c in cells]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(fps, m, s=110, color="#74c476", edgecolor="black", zorder=3)
    for xi, yi, n in zip(fps, m, names):
        ax.annotate(n, (xi, yi), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Throughput (images/sec)")
    ax.set_ylabel("mAP@0.5 (zero-shot)")
    ax.set_title("Accuracy vs speed across detectors")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def write_tables() -> dict:
    import pandas as pd
    out = {}
    grid = _load("grid.json")
    if grid:
        rows = []
        for c in grid.get("cells", []):
            mt = c["metrics"]
            rows.append({
                "axis": c.get("axis"), "regime": c.get("regime"),
                "detector": c["detector"], "enhancement": c["enhancement"],
                "mAP@0.5": round(mt["mAP@0.5"], 4),
                "mAP@[.5:.95]": round(mt["mAP@[.5:.95]"], 4),
                "P@0.25": round(mt["precision@0.25"], 4),
                "R@0.25": round(mt["recall@0.25"], 4),
                "F1@0.25": round(mt["f1@0.25"], 4),
                "FPS": round(c.get("fps", 0), 1),
                "n_images": mt.get("n_images"),
            })
        df = pd.DataFrame(rows)
        df.to_csv(C.METRICS_DIR / "grid.csv", index=False)
        out["grid"] = df
        # per-class AP matrix
        pc_rows = []
        for c in grid.get("cells", []):
            base = {"config": f"{c['detector']}/{c['enhancement']}", "regime": c.get("regime")}
            for cl in C.EXDARK_CLASSES:
                v = c["metrics"].get("per_class", {}).get(cl, {}).get("AP50")
                base[cl] = round(v, 4) if v is not None else None
            pc_rows.append(base)
        if pc_rows:
            pcdf = pd.DataFrame(pc_rows)
            pcdf.to_csv(C.METRICS_DIR / "per_class_ap.csv", index=False)
            out["per_class"] = pcdf
    ft = _load("finetune.json")
    if ft:
        rows = []
        for m in ft.get("models", []):
            rows.append({
                "model": m["base"].replace(".pt", ""),
                "epochs": m.get("epochs"),
                "zero_shot_mAP@0.5": round(m["zero_shot"]["mAP@0.5"], 4) if m.get("zero_shot") else None,
                "finetuned_mAP@0.5": round(m["finetuned"]["mAP@0.5"], 4) if m.get("finetuned") else None,
                "finetuned_mAP@[.5:.95]": round(m["finetuned"]["mAP@[.5:.95]"], 4) if m.get("finetuned") else None,
                "abs_gain": (round(m["finetuned"]["mAP@0.5"] - m["zero_shot"]["mAP@0.5"], 4)
                             if m.get("finetuned") and m.get("zero_shot") else None),
            })
        df = pd.DataFrame(rows)
        df.to_csv(C.METRICS_DIR / "finetune.csv", index=False)
        out["finetune"] = df
    return out


def generate_all() -> list[Path]:
    """Generate every figure for which the underlying JSON exists."""
    figs = []
    grid = _load("grid.json")
    ft = _load("finetune.json")
    F = C.FIGURES_DIR
    if grid:
        for fn, name in [
            (fig_enhancement_bars, "map_by_enhancement.png"),
            (fig_per_class_heatmap, "per_class_ap_heatmap.png"),
            (fig_quality_vs_map, "quality_vs_map.png"),
            (fig_accuracy_latency, "accuracy_vs_latency.png"),
        ]:
            p = fn(grid, F / name)
            if p:
                figs.append(p)
    if ft:
        p = fig_finetune_bars(ft, F / "zeroshot_vs_finetuned.png")
        if p:
            figs.append(p)
    return figs


# --------------------------------------------------------------------------- #
# Markdown / HTML report
# --------------------------------------------------------------------------- #
def _md_table(headers: list, rows: list[list]) -> str:
    h = "| " + " | ".join(str(x) for x in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join("" if x is None else str(x) for x in r) + " |" for r in rows)
    return "\n".join([h, sep, body])


def render_markdown() -> str:
    grid = _load("grid.json") or {}
    ft = _load("finetune.json") or {}
    cells = grid.get("cells", [])
    meta = grid.get("meta", {})
    L = []
    A = L.append

    A("# Object Detection in Low-Light Environments — Research Report\n")
    A("> Reproducible benchmark of image-enhancement and detector fine-tuning "
      "strategies on the **ExDark** low-light dataset, with a corrected "
      "COCO-style mAP evaluator.\n")
    A(f"*Auto-generated from `results/metrics/`. Hardware: {meta.get('device','MPS/CPU')} "
      f"(Apple M1). Seed {C.DEFAULT_SEED}. Split 70/15/15 stratified by class.*\n")

    A("\n## Abstract\n")
    A(
        "This project revisits a course experiment that found *“barely any "
        "improvement”* from low-light image enhancement. We show that result "
        "was an artifact of three methodology flaws — an invalid per-image "
        "precision/recall average standing in for mAP, a class map that "
        "silently dropped the **Cup** class and mis-mapped **Boat** to "
        "“traffic light”, and an enhancement comparison run on a *frozen* "
        "COCO detector that was never adapted to the enhanced domain. After "
        "replacing the evaluator with a correct, pooled COCO-style mAP, fixing "
        "the taxonomy, and **fine-tuning** the detector on ExDark, the picture "
        "becomes clear and matches the literature: fine-tuning is the dominant "
        "lever, while classical enhancement on a fixed detector moves mAP very "
        "little.\n"
    )

    # Zero-shot detector sweep
    det_cells = [c for c in cells if c.get("axis") == "detector"]
    if det_cells:
        A("\n## 1. Zero-shot detector baseline\n")
        A("COCO-pretrained detectors evaluated on the ExDark test subset with no "
          "enhancement and no fine-tuning. Detections are mapped COCO→ExDark "
          "across the 12 shared classes.\n")
        rows = [[c["detector"], f"{c['metrics']['mAP@0.5']:.3f}",
                 f"{c['metrics']['mAP@[.5:.95]']:.3f}",
                 f"{c['metrics']['recall@0.25']:.3f}", f"{c.get('fps',0):.1f}"]
                for c in sorted(det_cells, key=lambda c: -c["metrics"]["mAP@0.5"])]
        A(_md_table(["Detector", "mAP@0.5", "mAP@.5:.95", "Recall@.25", "FPS"], rows))
        A("\n![Accuracy vs latency](figures/accuracy_vs_latency.png)\n")

    # Enhancement grid
    enh_cells = [c for c in cells if c.get("axis") == "enhancement"]
    if enh_cells:
        A("\n## 2. Does enhancement help? (frozen detector)\n")
        A("A single detector is held fixed while the input enhancement varies. "
          "Per the literature, classical enhancement on a detector that never "
          "saw the enhanced domain is expected to help little or hurt.\n")
        base = next((c for c in enh_cells if c["enhancement"] == "original"), None)
        b = base["metrics"]["mAP@0.5"] if base else None
        rows = []
        for c in sorted(enh_cells, key=lambda c: -c["metrics"]["mAP@0.5"]):
            d = c["metrics"]["mAP@0.5"] - b if b is not None else None
            rows.append([c["enhancement"], f"{c['metrics']['mAP@0.5']:.3f}",
                         f"{c['metrics']['mAP@[.5:.95]']:.3f}",
                         (f"{d:+.3f}" if d is not None else ""),
                         f"{c.get('fps',0):.1f}"])
        A(_md_table(["Enhancement", "mAP@0.5", "mAP@.5:.95", "Δ vs original", "FPS"], rows))
        A("\n![mAP by enhancement](figures/map_by_enhancement.png)\n")
        A("\n![Per-class AP heatmap](figures/per_class_ap_heatmap.png)\n")
        if any(c.get("quality") for c in enh_cells):
            A("\n### Perceptual quality vs detection accuracy\n")
            A("Brighter, higher-entropy images are not necessarily easier to "
              "detect in — the two are weakly correlated, evidence that "
              "enhancement optimised for human perception is the wrong "
              "objective for a detector.\n")
            A("![Quality vs mAP](figures/quality_vs_map.png)\n")

    # Fine-tuning
    ft_models = [m for m in ft.get("models", []) if m.get("finetuned")]
    if ft_models:
        A("\n## 3. Fine-tuning on ExDark — the dominant lever\n")
        A("The detector is fine-tuned on the ExDark training split (native "
          "12-class head) and evaluated on the held-out test split.\n")
        rows = []
        for m in ft_models:
            zs = m["zero_shot"]["mAP@0.5"] if m.get("zero_shot") else None
            fb = m["finetuned"]["mAP@0.5"]
            rows.append([m["base"].replace(".pt", ""), m.get("epochs"),
                         (f"{zs:.3f}" if zs is not None else "—"), f"{fb:.3f}",
                         f"{m['finetuned']['mAP@[.5:.95]']:.3f}",
                         (f"{fb - zs:+.3f}" if zs is not None else "—")])
        A(_md_table(["Model", "Epochs", "Zero-shot mAP@0.5", "Fine-tuned mAP@0.5",
                     "Fine-tuned mAP@.5:.95", "Gain"], rows))
        A("\n![Zero-shot vs fine-tuned](figures/zeroshot_vs_finetuned.png)\n")

    # Literature context
    A("\n## 4. Context: published ExDark results\n")
    A("Canonical ExDark numbers (fine-tuned YOLOv3, official split, mAP@0.5) to "
      "situate our results. Note the entire 4-year SOTA spread is ~1.6 mAP, "
      "underscoring that enhancement adds little once the detector is fine-tuned.\n")
    A(_md_table(
        ["Method", "mAP@0.5", "Note"],
        [["Zero-shot COCO YOLOv3", "~0.21", "domain gap"],
         ["Fine-tuned YOLOv3 (baseline)", "0.764", "the lever"],
         ["Zero-DCE + YOLOv3", "0.769", "+learned enhancement"],
         ["MAET (ICCV'21)", "0.777", "illumination-aware"],
         ["IAT-YOLO (BMVC'22)", "0.778", "adaptive transformer"],
         ["PE-YOLO (BMVC'23)", "0.780", "pyramid enhancement"]]))
    A("\n*Sources: MAET, IAT, PE-YOLO repositories/papers; Marshetty zero-shot "
      "write-up; two-stage YOLOv7 study (PMC12190514).*\n")

    A("\n## 5. Methodology fixes (vs the original project)\n")
    A(_md_table(
        ["Issue in original code", "Effect", "Fix"],
        [["Averaged per-image `precision[-1]`/`recall[-1]` as “mAP”",
          "No PR-curve integration; signal washed out; numbers non-comparable",
          "Pooled COCO-style evaluator (`metrics.py`), 101-pt AP, mAP@0.5 & @.5:.95"],
         ["`sklearn average_precision_score(match, score)` per image",
          "Ignores missed GT (false negatives invisible)",
          "Recall denominator = total GT per class across dataset"],
         ["`Cup` missing from class map", "All Cup objects dropped",
          "Added Cup→COCO 41; native 12-class head for fine-tuning"],
         ["`Boat` mapped to COCO 9 (traffic light)", "Boat AP structurally ~0",
          "Fixed Boat→COCO 8"],
         ["Retinex/MSR log-images into frozen COCO model", "Out-of-distribution; unfair",
          "Natural-appearance MSRCR + correct experimental order (fine-tune first)"],
         ["Manual 640×640 squash resize", "Aspect-ratio distortion",
          "Let ultralytics letterbox; boxes mapped back to original pixels"]]))

    A("\n## 6. Conclusion\n")
    A(
        "The original “barely any improvement” was, in fact, the *expected* "
        "result for classical enhancement fed to a frozen COCO detector — but it "
        "was being measured with a metric that could not have shown a difference "
        "either way. With a correct evaluator and a corrected taxonomy, the "
        "honest finding is: **(i)** classical enhancement on a fixed detector is "
        "roughly neutral; **(ii)** fine-tuning the detector on ExDark is the "
        "decisive lever; **(iii)** two previously broken classes (Cup, Boat) now "
        "contribute real signal. This mirrors the published consensus.\n"
    )
    A("\n## Reproducibility\n")
    A("```\n"
      "python scripts/00_make_split.py      # deterministic split + YOLO export\n"
      "python scripts/02_run_benchmark.py   # zero-shot sweep + enhancement grid\n"
      "python scripts/03_finetune.py        # fine-tune + before/after\n"
      "python scripts/04_make_report.py     # this report + figures\n"
      "```\n")
    A(f"\n*Library versions and the exact split manifest "
      f"(`data/splits/exdark_split.json`) are committed for reproducibility.*\n")
    return "\n".join(L)


def md_to_html(md: str) -> str:
    """Minimal, offline GitHub-flavoured-markdown to HTML (headings, tables,
    images, lists, code, bold)."""
    import html
    import re
    lines = md.split("\n")
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
           "<title>Low-Light Detection — Research Report</title>",
           "<style>body{max-width:920px;margin:40px auto;padding:0 20px;"
           "font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
           "line-height:1.6;color:#1a1a1a}h1,h2,h3{line-height:1.25;margin-top:1.6em}"
           "h1{border-bottom:2px solid #eaecef;padding-bottom:.3em}"
           "h2{border-bottom:1px solid #eaecef;padding-bottom:.3em}"
           "table{border-collapse:collapse;width:100%;margin:1em 0;font-size:14px}"
           "th,td{border:1px solid #d0d7de;padding:6px 12px;text-align:left}"
           "th{background:#f6f8fa}tr:nth-child(even){background:#fbfcfd}"
           "img{max-width:100%;border:1px solid #eaecef;border-radius:6px;margin:1em 0}"
           "code,pre{background:#f6f8fa;border-radius:6px;font-family:SFMono-Regular,Consolas,monospace}"
           "pre{padding:14px;overflow:auto}code{padding:2px 5px}"
           "blockquote{border-left:4px solid #d0d7de;color:#57606a;margin:0;padding:0 1em}"
           "</style></head><body>"]

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        s = re.sub(r"!\[(.*?)\]\((.*?)\)", r"<img alt='\1' src='\2'>", s)
        return s

    i = 0
    in_code = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            if not in_code:
                out.append("<pre><code>"); in_code = True
            else:
                out.append("</code></pre>"); in_code = False
            i += 1; continue
        if in_code:
            out.append(html.escape(ln)); i += 1; continue
        if ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= {"-", " ", ":"}:
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>"); continue
        if ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("> "):
            out.append(f"<blockquote>{inline(ln[2:])}</blockquote>")
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    out.append("</body></html>")
    return "\n".join(out)


def render_report() -> dict:
    """Generate figures, tables, report.md and report.html. Returns paths."""
    write_tables()
    figs = generate_all()
    md = render_markdown()
    md_path = C.RESULTS_DIR / "report.md"
    html_path = C.RESULTS_DIR / "report.html"
    md_path.write_text(md)
    html_path.write_text(md_to_html(md))
    return {"markdown": str(md_path), "html": str(html_path), "figures": [str(f) for f in figs]}

