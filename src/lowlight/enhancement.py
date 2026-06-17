"""Low-light image enhancement methods, exposed through a single registry.

Every method takes a BGR ``uint8`` image and returns a BGR ``uint8`` image, so
they are interchangeable in the benchmark. The registry (:data:`ENHANCERS`)
maps a short name to a callable.

Design notes
------------
* The original project applied histogram equalisation / CLAHE on the **V**
  channel of HSV. We additionally provide CLAHE on the **L** channel of LAB,
  which is the more standard, less colour-distorting choice.
* The original Retinex/MSR returned the raw log-difference image. Fed to a
  COCO-pretrained detector that expects natural photographs, these grey,
  washed-out images hurt detection. We add **MSRCR** (Multi-Scale Retinex with
  Color Restoration) and a gain/offset + simple colour-restoration step, which
  produces far more natural output — the principled version of Retinex for a
  downstream detector.
* ``adaptive_gamma`` chooses gamma from image mean brightness, instead of a
  fixed constant, so dark and very-dark images are treated differently.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Simple point operations
# --------------------------------------------------------------------------- #
def identity(image: np.ndarray) -> np.ndarray:
    """No enhancement (baseline)."""
    return image


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)


def adaptive_gamma(image: np.ndarray) -> np.ndarray:
    """Gamma chosen from mean luminance: darker image -> stronger brightening."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean()) / 255.0
    mean = min(max(mean, 1e-3), 0.5)
    # Solve out_mean = mean**(1/gamma) = 0.5  ->  gamma = log(mean)/log(0.5).
    # Darker image (smaller mean) -> larger gamma -> stronger brightening.
    gamma = float(np.clip(np.log(mean) / np.log(0.5), 1.0, 3.0))
    return gamma_correction(image, gamma)


def brightness_contrast(image: np.ndarray, brightness: int = 30, contrast: int = 30) -> np.ndarray:
    alpha = 1.0 + (contrast / 127.0)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=brightness)


# --------------------------------------------------------------------------- #
# Histogram-based
# --------------------------------------------------------------------------- #
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global HE on the value channel (HSV)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def clahe_hsv(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the HSV value channel (matches the original project)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def clahe_lab(image: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB (less colour distortion than HSV-V)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# Retinex family
# --------------------------------------------------------------------------- #
def _single_scale_retinex(img_float: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)
    return np.log10(img_float) - np.log10(blur + 1e-6)


def single_scale_retinex(image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    img = image.astype(np.float32) + 1.0
    r = _single_scale_retinex(img, sigma)
    return _stretch_uint8(r)


def multi_scale_retinex(image: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    img = image.astype(np.float32) + 1.0
    r = np.zeros_like(img)
    for s in sigmas:
        r += _single_scale_retinex(img, s)
    r /= len(sigmas)
    return _stretch_uint8(r)


def msrcr(
    image: np.ndarray,
    sigmas=(15, 80, 250),
    alpha: float = 125.0,
    beta: float = 46.0,
    gain: float = 192.0,
    offset: float = -30.0,
) -> np.ndarray:
    """Multi-Scale Retinex with Color Restoration (Jobson et al., 1997).

    Produces natural-looking, colour-faithful output — unlike raw MSR, which
    returns a grey log-difference image. This is the Retinex variant suited to
    feeding a detector trained on natural photographs.
    """
    img = image.astype(np.float32) + 1.0
    # Multi-scale retinex
    msr = np.zeros_like(img)
    for s in sigmas:
        msr += _single_scale_retinex(img, s)
    msr /= len(sigmas)
    # Colour restoration term
    intensity = np.sum(img, axis=2, keepdims=True)
    cr = beta * (np.log10(alpha * img) - np.log10(intensity))
    out = gain * (msr * cr) + offset
    # Per-channel simple white-balance stretch (clip 1%/99%)
    return _stretch_uint8(out, clip_percent=1.0)


# --------------------------------------------------------------------------- #
# White balance + dehazing-style
# --------------------------------------------------------------------------- #
def gray_world(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance followed by a mild adaptive gamma."""
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = means.mean()
    img = np.clip(img * (gray / means), 0, 255).astype(np.uint8)
    return adaptive_gamma(img)


def dark_channel_lowlight(image: np.ndarray, omega: float = 0.85, t0: float = 0.1) -> np.ndarray:
    """Low-light enhancement via the inverted-image dark-channel-prior trick.

    Inverting a low-light image makes it resemble a hazy image; applying dark
    channel prior dehazing and inverting back brightens it while preserving
    structure (Dong et al., 2011).
    """
    inv = 255 - image
    f = inv.astype(np.float32) / 255.0
    # dark channel
    minc = np.min(f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(minc, kernel)
    # atmospheric light: brightest 0.1% in dark channel
    flat = dark.ravel()
    n = max(int(flat.size * 0.001), 1)
    idx = np.argpartition(flat, -n)[-n:]
    A = f.reshape(-1, 3)[idx].max(axis=0)
    A = np.maximum(A, 1e-3)
    # transmission
    t = 1 - omega * cv2.erode(np.min(f / A, axis=2), kernel)
    t = np.clip(t, t0, 1.0)[..., None]
    j = (f - A) / t + A
    j = np.clip(j, 0, 1)
    out = 255 - (j * 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _stretch_uint8(arr: np.ndarray, clip_percent: float = 0.0) -> np.ndarray:
    """Stretch a float array to [0,255] uint8, optionally clipping percentiles."""
    out = arr.astype(np.float32)
    if clip_percent > 0:
        lo = np.percentile(out, clip_percent)
        hi = np.percentile(out, 100 - clip_percent)
    else:
        lo, hi = out.min(), out.max()
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (out - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ENHANCERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": identity,
    "gamma": lambda im: gamma_correction(im, 1.8),
    "adaptive_gamma": adaptive_gamma,
    "brightness_contrast": lambda im: brightness_contrast(im, 30, 30),
    "hist_eq": histogram_equalization,
    "clahe_hsv": clahe_hsv,
    "clahe_lab": clahe_lab,
    "ssr": single_scale_retinex,
    "msr": multi_scale_retinex,
    "msrcr": msrcr,
    "gray_world": gray_world,
    "dcp": dark_channel_lowlight,
}

# A focused default set used for the headline benchmark grid (keeps wall-clock
# reasonable while covering each family: point-op, histogram, retinex, WB/DCP).
DEFAULT_METHODS = [
    "original", "gamma", "adaptive_gamma", "hist_eq",
    "clahe_hsv", "clahe_lab", "msr", "msrcr", "gray_world", "dcp",
]


def get(name: str) -> Callable[[np.ndarray], np.ndarray]:
    return ENHANCERS[name]
