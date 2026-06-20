"""Domain fine-tuning A/B: does enhancement help once the detector *adapts*?

Every other experiment in this repo uses a **frozen** COCO detector, where the
literature (and our own benchmark) says classical enhancement is ~neutral. The
hypothesis "enhancement lets the model recognise more" can only be tested fairly
by letting the detector *learn* the enhanced domain. This module does exactly
that, as a controlled A/B:

    for E in {original, clahe_lab}:
        export a YOLO dataset whose images are enhanced with E
        fine-tune the SAME base detector on it (same epochs/imgsz/split/seed)
    evaluate every trained model on every test domain (a 2x2 matrix)

The **matched diagonal** is the decisive comparison: train-and-test-on-original
vs train-and-test-on-enhanced. The off-diagonal quantifies train/test domain
shift. Everything is held constant except the enhancement, so any mAP gap is
attributable to enhancement — the apples-to-apples test the frozen benchmark
cannot provide.

Designed for a CUDA GPU (training the full split on CPU is impractical). Results
land in ``results/metrics/domain_finetune.json`` and ``results/domain.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from . import config as C
from . import dataset as D
from . import enhancement as Enh
from . import finetune as FT


def export_enhanced_yolo(enhancer: str, split: dict, *, out_dir: Path | None = None,
                         overwrite: bool = False) -> Path:
    """Write a YOLO dataset whose images are pre-enhanced with ``enhancer``.

    ``original`` reuses the standard (symlinked) export — no pixels are rewritten.
    Any other enhancer materialises actual enhanced image files (so training and
    val() both see the enhanced domain) with the identical labels and split.
    Returns the dataset.yaml path.
    """
    if enhancer == "original":
        return FT.ensure_export()

    out_dir = out_dir or (C.DATA_DIR / f"exdark_yolo_{enhancer}")
    yaml_path = out_dir / "dataset.yaml"
    if yaml_path.exists() and not overwrite:
        return yaml_path

    fn = Enh.get(enhancer)
    for sub in ("images", "labels"):
        for part in ("train", "val", "test"):
            (out_dir / sub / part).mkdir(parents=True, exist_ok=True)

    n_written = 0
    for part, items in split.items():
        if part not in ("train", "val", "test"):
            continue
        for s in items:
            img = cv2.imread(s.image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            objs = D.parse_bbgt(s.annot_path, clip_to=(w, h))
            if not objs:
                continue
            try:
                out = fn(img)
            except Exception:
                out = img
            # Always write JPEG so the on-disk domain is well-defined.
            dst_img = out_dir / "images" / part / f"{s.stem}.jpg"
            dst_lbl = out_dir / "labels" / part / f"{s.stem}.txt"
            cv2.imwrite(str(dst_img), out)
            with open(dst_lbl, "w") as f:
                for o in objs:
                    cx = ((o.x1 + o.x2) / 2) / w
                    cy = ((o.y1 + o.y2) / 2) / h
                    bw = (o.x2 - o.x1) / w
                    bh = (o.y2 - o.y1) / h
                    f.write(f"{o.exdark_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            n_written += 1

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(C.EXDARK_CLASSES))
    yaml_path.write_text(
        f"# ExDark pre-enhanced with '{enhancer}' (auto-generated)\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(C.EXDARK_CLASSES)}\nnames:\n{names_block}\n"
    )
    print(f"    exported {n_written} '{enhancer}'-enhanced images -> {out_dir}")
    return yaml_path


def run_domain_experiment(base="yolov8n.pt", enhancers=("original", "clahe_lab"), *,
                          epochs=100, imgsz=640, device=None, fraction=1.0,
                          batch=16, freeze=None, cross_eval=True) -> dict:
    """Fine-tune ``base`` once per enhancer and cross-evaluate. Returns a record."""
    device = device or C.pick_device()
    split = D.load_split(C.DATA_DIR / "splits" / "exdark_split.json")

    # 1) Ensure a YOLO export exists for each enhancement domain.
    yamls = {}
    for e in enhancers:
        print(f"[export] domain '{e}' ...")
        yamls[e] = str(export_enhanced_yolo(e, split))

    # 2) Fine-tune one model per enhancement domain (everything else fixed).
    models = {}
    for e in enhancers:
        print(f"\n[train] base={base} on '{e}' domain "
              f"(epochs={epochs}, imgsz={imgsz}, fraction={fraction}, device={device}) ...")
        art = FT.finetune(base=base, epochs=epochs, imgsz=imgsz, batch=batch,
                          device=device, fraction=fraction, freeze=freeze,
                          data_yaml=yamls[e], name=f"domain_{Path(base).stem}_{e}")
        models[e] = art["best"]
        print(f"        -> {art['best']}")

    # 3) Evaluate each model on each test domain (2x2 if cross_eval else diagonal).
    test_domains = list(enhancers) if cross_eval else None
    results = []
    for train_e, weights in models.items():
        eval_domains = enhancers if cross_eval else [train_e]
        for test_e in eval_domains:
            print(f"[eval] train='{train_e}'  test='{test_e}' ...")
            m = FT.evaluate_on_test(weights, device=device, split="test",
                                    imgsz=imgsz, data_yaml=yamls[test_e])
            results.append({"train_domain": train_e, "test_domain": test_e,
                            "matched": train_e == test_e,
                            "mAP@0.5": m["mAP@0.5"], "mAP@[.5:.95]": m["mAP@[.5:.95]"],
                            "precision": m["precision"], "recall": m["recall"],
                            "per_class_AP50": m["per_class_AP50"], "weights": weights})
            print(f"       mAP@0.5={m['mAP@0.5']:.3f}  mAP@.5:.95={m['mAP@[.5:.95]']:.3f}")

    record = {"base": base, "epochs": epochs, "imgsz": imgsz, "fraction": fraction,
              "device": device, "enhancers": list(enhancers), "models": models,
              "results": results}
    out = C.METRICS_DIR / "domain_finetune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    write_domain_md(record)
    print(f"\nWrote {out} and {C.RESULTS_DIR / 'domain.md'}")
    return record


def write_domain_md(record: dict) -> Path:
    enhs = record["enhancers"]
    res = {(r["train_domain"], r["test_domain"]): r for r in record["results"]}
    L, A = [], None
    A = L.append
    A("# Domain Fine-Tuning: does enhancement help once the detector adapts?\n")
    A(f"*Base `{record['base']}` · {record['epochs']} epochs · imgsz {record['imgsz']} "
      f"· fraction {record['fraction']} · device `{record['device']}` · seed {C.DEFAULT_SEED}.*\n")
    A("Each model is fine-tuned on one enhancement domain and evaluated on each "
      "test domain. Only the image enhancement differs — base, epochs, split, and "
      "seed are identical — so any mAP gap is attributable to enhancement.\n")

    # 2x2 (or diagonal) mAP@0.5 matrix: rows = train domain, cols = test domain
    A("## mAP@0.5 — train domain (rows) × test domain (columns)\n")
    A("| train ↓ / test → | " + " | ".join(enhs) + " |")
    A("| --- | " + " | ".join("---" for _ in enhs) + " |")
    for tr in enhs:
        cells = []
        for te in enhs:
            r = res.get((tr, te))
            cells.append(f"{r['mAP@0.5']:.3f}" + (" *(matched)*" if tr == te else "") if r else "—")
        A(f"| **{tr}** | " + " | ".join(cells) + " |")
    A("")

    # The headline: matched-diagonal comparison
    A("## Headline — matched train+test (the fair A/B)\n")
    A("| domain | mAP@0.5 | mAP@.5:.95 | precision | recall |")
    A("| --- | --- | --- | --- | --- |")
    diag = {e: res.get((e, e)) for e in enhs}
    for e in enhs:
        r = diag[e]
        if r:
            A(f"| {e} | {r['mAP@0.5']:.3f} | {r['mAP@[.5:.95]']:.3f} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} |")
    base_e = enhs[0]
    if diag.get(base_e):
        b = diag[base_e]["mAP@0.5"]
        A("")
        for e in enhs[1:]:
            if diag.get(e):
                d = diag[e]["mAP@0.5"] - b
                verdict = ("**enhancement helps**" if d > 0.005 else
                           "**enhancement hurts**" if d < -0.005 else
                           "**no meaningful difference**")
                A(f"- `{e}` vs `{base_e}`: **{d:+.3f} mAP@0.5** → {verdict}")
    A("")

    # Per-class diagonal comparison (the "recognize more classes" question)
    if len(enhs) >= 2 and diag.get(enhs[0]) and diag.get(enhs[1]):
        a, b = enhs[0], enhs[1]
        pa = diag[a]["per_class_AP50"]; pb = diag[b]["per_class_AP50"]
        A(f"## Per-class AP@0.5 — matched `{a}` vs matched `{b}`\n")
        A(f"| class | {a} | {b} | Δ |")
        A("| --- | --- | --- | --- |")
        for c in sorted(set(pa) | set(pb), key=lambda c: -(pb.get(c, 0) - pa.get(c, 0))):
            va, vb = pa.get(c, 0.0), pb.get(c, 0.0)
            A(f"| {c} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} |")
        A("")
        A("> A positive Δ on a class means the enhanced-domain model recognises "
          "that class better than the original-domain model — direct evidence for "
          "(or against) 'enhancement → recognise more'.\n")

    out = C.RESULTS_DIR / "domain.md"
    out.write_text("\n".join(L))
    return out
