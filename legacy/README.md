# Legacy code (original course project)

These files are the **original** implementation, preserved for provenance. They
are the subject of the bug analysis in [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md)
and are **superseded** by the `src/lowlight/` package.

- `object_detection.py`, `enhance.py` — original frozen-COCO + classical-enhancement
  pipeline with the per-image `precision[-1]`/`recall[-1]` metric, the dropped
  `Cup` class, and `Boat → 9` (traffic light) bug.
- `test_msr.py`, `test_msr_simple.py` — ad-hoc Multi-Scale Retinex smoke tests
  (replaced by `tests/`).
- `notebooks/` — early experimental scratch scripts.

Use the new framework instead — see the top-level [`README.md`](../README.md).
