"""Cross-model x enhancement comparison.

The headline question of this project is: *does image enhancement let a detector
recognise more, and does that hold across different detectors?* The base
``benchmark`` module sweeps detectors (enhancement fixed) and enhancements
(detector fixed) on separate axes. This module joins them into a single
**detector x enhancement matrix** of mAP@0.5 so every model is compared against
every enhancement — and, critically, against its own ``original`` (no-enhancement)
baseline.

One inference pass per (detector, enhancement) cell feeds three artifacts:

  * ``results/metrics/model_enhancement_matrix.{json,csv}`` — the raw matrix.
  * ``results/figures/model_enhancement_matrix.png``        — a heatmap.
  * ``results/comparison.md``                               — a plain-language
    summary: best enhancement per model, and delta vs the no-enhancement default.

It also emits standard ``grid.json`` cells (detector-axis from each model's
``original`` column; enhancement-axis from the primary detector) so the existing
``report`` module keeps working from the same single compute pass.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from . import detectors as Det
from . import runner as R


def run_matrix(detectors, enhancers, samples, *, device="cpu", imgsz=C.DEFAULT_IMGSZ,
               conf=C.DEFAULT_CONF, primary=None, progress=True, n_boot=100):
    """Compute a detector x enhancement matrix of metrics in one pass.

    Each model is loaded once and run against every enhancement over ``samples``.
    Returns ``(matrix, cells)`` where ``matrix[detector][enhancement]`` is a dict
    of metrics and ``cells`` is a flat list of cell dicts (grid.json schema).

    ``n_boot`` controls the bootstrap CI resamples (0 disables CIs for speed).
    """
    primary = primary or (detectors[0] if detectors else None)
    matrix: dict[str, dict[str, dict]] = {}
    cells: list[dict] = []

    for det_name in detectors:
        try:
            model = Det.load_detector(det_name)
        except Exception as e:  # missing weights / download failure -> skip, keep going
            print(f"  skip detector {det_name}: {e}")
            continue
        coco_space = Det.is_coco_model(model)
        matrix[det_name] = {}
        print(f"\n== {det_name} ({'COCO 80-class' if coco_space else 'native 12-class'}) ==")
        for enh in enhancers:
            res = R.run_cell(model, det_name, enh, samples, device=device, imgsz=imgsz,
                             conf=conf, coco_space=coco_space, keep_per_image=(n_boot > 0),
                             progress=progress)
            ci = R.bootstrap_map50_ci(res.per_image, n_boot=n_boot) if n_boot > 0 else (float("nan"), float("nan"))
            matrix[det_name][enh] = {
                "mAP@0.5": res.metrics["mAP@0.5"],
                "mAP@[.5:.95]": res.metrics["mAP@[.5:.95]"],
                "recall@0.25": res.metrics.get("recall@0.25"),
                "fps": res.fps,
                "ci95_map50": ci,
                "n_images": res.n_images,
                "per_class": {k: v["AP50"] for k, v in res.metrics["per_class"].items()},
            }
            cell = {
                "axis": "enhancement" if det_name == primary else "matrix",
                "regime": "frozen-coco" if coco_space else "finetuned",
                "detector": det_name,
                "enhancement": enh,
                "metrics": res.metrics,
                "sec_per_img": res.sec_per_img,
                "fps": res.fps,
                "ci95_map50": ci,
                "quality": None,
            }
            cells.append(cell)
            # The 'original' column doubles as the zero-shot detector-sweep cell.
            if enh == "original":
                cells.append({**cell, "axis": "detector"})
            m = matrix[det_name][enh]
            print(f"  {enh:18s} mAP@0.5={m['mAP@0.5']:.3f}  "
                  f"mAP@.5:.95={m['mAP@[.5:.95]']:.3f}  FPS={m['fps']:.1f}")
    return matrix, cells


def write_matrix_artifacts(matrix: dict, *, out_metrics=C.METRICS_DIR, out_figures=C.FIGURES_DIR,
                           out_md=C.RESULTS_DIR / "comparison.md", meta=None) -> dict:
    """Write the matrix JSON/CSV, a heatmap, and a plain-language comparison.md."""
    out_metrics, out_figures = Path(out_metrics), Path(out_figures)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    detectors = list(matrix.keys())
    enhancers: list[str] = []
    for d in detectors:
        for e in matrix[d]:
            if e not in enhancers:
                enhancers.append(e)

    json_path = out_metrics / "model_enhancement_matrix.json"
    json_path.write_text(json.dumps({"meta": meta or {}, "matrix": matrix}, indent=2))

    # CSV: rows = detector, columns = enhancement, values = mAP@0.5
    csv_path = out_metrics / "model_enhancement_matrix.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector"] + enhancers + ["best_enh", "best_mAP@0.5", "delta_vs_original"])
        for d in detectors:
            row = [d]
            vals = {}
            for e in enhancers:
                v = matrix[d].get(e, {}).get("mAP@0.5")
                vals[e] = v
                row.append(f"{v:.4f}" if v is not None else "")
            best_e = max((e for e in enhancers if vals.get(e) is not None),
                         key=lambda e: vals[e], default=None)
            orig = vals.get("original")
            best_v = vals.get(best_e) if best_e else None
            delta = (best_v - orig) if (best_v is not None and orig is not None) else None
            row += [best_e or "",
                    f"{best_v:.4f}" if best_v is not None else "",
                    f"{delta:+.4f}" if delta is not None else ""]
            w.writerow(row)

    fig_path = _heatmap(matrix, detectors, enhancers, out_figures / "model_enhancement_matrix.png")
    md_path = _comparison_md(matrix, detectors, enhancers, out_md, meta=meta)
    return {"json": str(json_path), "csv": str(csv_path),
            "figure": str(fig_path) if fig_path else None, "markdown": str(md_path)}


def _heatmap(matrix, detectors, enhancers, out: Path):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    grid = np.full((len(detectors), len(enhancers)), np.nan)
    for i, d in enumerate(detectors):
        for j, e in enumerate(enhancers):
            v = matrix[d].get(e, {}).get("mAP@0.5")
            if v is not None:
                grid[i, j] = v
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(enhancers)), max(3, 0.8 * len(detectors)) + 1))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(enhancers)))
    ax.set_xticklabels(enhancers, rotation=45, ha="right")
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors)
    for i in range(len(detectors)):
        for j in range(len(enhancers)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < np.nanmean(grid) else "black", fontsize=8)
    ax.set_title("mAP@0.5 — detector x enhancement")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mAP@0.5")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _comparison_md(matrix, detectors, enhancers, out: Path, meta=None):
    L = []
    A = L.append
    A("# Model × Enhancement Comparison\n")
    if meta:
        A(f"*Auto-generated. Device: `{meta.get('device')}` · "
          f"{meta.get('subset_per_class')} images/class · seed {C.DEFAULT_SEED}.*\n")
    A("Every detector is evaluated against every enhancement on the **same** "
      "stratified test subset, with the **same** pooled COCO-style evaluator. "
      "The `original` column is each model's no-enhancement baseline; `Δ` is the "
      "best enhancement minus that baseline.\n")

    # Main matrix table (mAP@0.5)
    A("## mAP@0.5 matrix\n")
    A("| Detector | " + " | ".join(enhancers) + " |")
    A("| --- | " + " | ".join("---" for _ in enhancers) + " |")
    for d in detectors:
        cells = []
        orig = matrix[d].get("original", {}).get("mAP@0.5")
        best = max((matrix[d][e]["mAP@0.5"] for e in matrix[d]), default=None)
        for e in enhancers:
            v = matrix[d].get(e, {}).get("mAP@0.5")
            if v is None:
                cells.append("")
            else:
                mark = "**" if v == best else ""
                cells.append(f"{mark}{v:.3f}{mark}")
        A(f"| {d} | " + " | ".join(cells) + " |")
    A("")

    # Per-model verdict
    A("## Does enhancement help each model?\n")
    A("| Detector | original | best enhancement | best mAP@0.5 | Δ vs original |")
    A("| --- | --- | --- | --- | --- |")
    for d in detectors:
        vals = {e: matrix[d][e]["mAP@0.5"] for e in matrix[d]}
        orig = vals.get("original")
        best_e = max(vals, key=vals.get) if vals else None
        best_v = vals.get(best_e) if best_e else None
        delta = (best_v - orig) if (best_v is not None and orig is not None) else None
        A(f"| {d} | {orig:.3f} | {best_e} | {best_v:.3f} | "
          f"{delta:+.3f} |" if delta is not None else f"| {d} | - | - | - | - |")
    A("")

    # Best enhancement overall, averaged across models
    A("## Which enhancement is best on average (across all models)?\n")
    avg = {}
    for e in enhancers:
        xs = [matrix[d][e]["mAP@0.5"] for d in detectors if e in matrix[d]]
        if xs:
            avg[e] = sum(xs) / len(xs)
    orig_avg = avg.get("original")
    A("| Enhancement | mean mAP@0.5 | Δ vs original |")
    A("| --- | --- | --- |")
    for e, v in sorted(avg.items(), key=lambda kv: -kv[1]):
        d = (v - orig_avg) if orig_avg is not None else None
        A(f"| {e} | {v:.3f} | {d:+.3f} |" if d is not None else f"| {e} | {v:.3f} | - |")
    A("")

    A("![Heatmap](figures/model_enhancement_matrix.png)\n")
    A("> **How to read this.** If the best-enhancement column barely beats "
      "`original` (small positive Δ, often within the bootstrap CI), the honest "
      "conclusion is that classical enhancement on a *frozen* COCO detector is "
      "roughly neutral — adapting the detector (fine-tuning) is the real lever. "
      "See `results/report.md` for the fine-tuning arm.\n")

    out.write_text("\n".join(L))
    return out
