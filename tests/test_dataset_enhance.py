"""Tests for dataset taxonomy and enhancement output invariants."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lowlight import config as C  # noqa: E402
from lowlight import enhancement as E  # noqa: E402


def test_taxonomy_complete_and_fixed():
    # All 12 classes present, Cup included, Boat correctly mapped to COCO 8.
    assert len(C.EXDARK_CLASSES) == 12
    assert "Cup" in C.EXDARK_TO_COCO and C.EXDARK_TO_COCO["Cup"] == 41
    assert C.EXDARK_TO_COCO["Boat"] == 8  # not 9 (traffic light)
    assert len(C.EXDARK_TO_COCO) == 12


def test_enhancers_preserve_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 60).astype(np.uint8)  # dark-ish
    for name in E.DEFAULT_METHODS:
        out = E.get(name)(img.copy())
        assert out.dtype == np.uint8, name
        assert out.shape == img.shape, (name, out.shape)


def test_adaptive_gamma_brightens_dark_image():
    img = np.full((64, 64, 3), 18, np.uint8)  # very dark
    out = E.adaptive_gamma(img)
    assert out.mean() > img.mean() + 20


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
