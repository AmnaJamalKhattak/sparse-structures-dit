"""Where in an image an intervention landed: composition, or contours and texture?

The register ablations in this project produce small but systematic image changes, and
the open question is *what kind* of change.  A global pixel distance cannot tell a
recomposed scene from a scene with softer edges, so these three readouts split the
difference map by spatial frequency and by where the clean image has structure.

Deliberately elementary.  Every quantity here is a mean of squares over an explicit
mask, computed with numpy and no learned model, so a reviewer can re-derive any number
from the saved images.  Nothing is calibrated, fitted or thresholded against an outcome.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# Radial band edges as a fraction of the Nyquist radius.  Thirds, because any other
# split invites the suspicion that the boundaries were chosen once the answer was known.
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("low", 0.0, 1.0 / 3.0),
    ("mid", 1.0 / 3.0, 2.0 / 3.0),
    ("high", 2.0 / 3.0, 1.0),
)


def to_gray(image) -> Optional[np.ndarray]:
    """Luminance in [0, 255] as float32, or ``None`` when there is no image.

    Rec. 601 luma, which is what PIL's ``convert("L")`` uses, so a reader comparing
    these numbers against a greyscale export of the same PNG gets the same array.
    """
    if image is None:
        return None
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] >= 3:
        return (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2])
    return None


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude from the 3x3 Sobel pair, same shape as the input.

    Implemented by shifted slices rather than a convolution library so this module
    needs nothing beyond numpy.  Edge pixels are replicated, which keeps the shape and
    biases nothing: every comparison here is between two images of identical size.
    """
    padded = np.pad(gray, 1, mode="edge")
    north_west, north, north_east = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    south_west, south, south_east = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    gx = (north_east + 2.0 * east + south_east) - (north_west + 2.0 * west + south_west)
    gy = (south_west + 2.0 * south + south_east) - (north_west + 2.0 * north + north_east)
    return np.sqrt(gx ** 2 + gy ** 2)


def _radial_bands(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Boolean masks over the fftshifted frequency plane, one per band."""
    rows, cols = shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    # Normalised so the Nyquist radius is 1.0 regardless of the image size, which is
    # what makes the bands comparable across resolutions.
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2) / np.sqrt(2.0)
    masks = {}
    for name, lo, hi in BANDS:
        inner = radius >= lo
        outer = radius < hi if name != "high" else radius <= hi + 1e-9
        masks[name] = inner & outer
    return masks


def spectrum_bands(clean, treated) -> Dict[str, float]:
    """How the difference map's energy splits across low, mid and high frequency.

    The fractions are of the *difference*, not of either image: the question is where
    the intervention's effect sits, so the clean image's own spectrum is not the
    denominator.  ``high_over_low`` is reported because it is the single number that
    separates "the scene changed" from "the detail changed" -- large means the effect
    is in texture and contours rather than in composition.
    """
    a, b = to_gray(clean), to_gray(treated)
    out: Dict[str, float] = {}
    for name, _, _ in BANDS:
        out[f"spectrum_{name}_fraction"] = float("nan")
        out[f"spectrum_{name}_area"] = float("nan")
        out[f"spectrum_{name}_over_uniform"] = float("nan")
    out["spectrum_high_over_low"] = float("nan")
    if a is None or b is None or a.shape != b.shape:
        return out
    masks = _radial_bands(a.shape)
    # The band areas are reported alongside the fractions because they are the
    # white-noise expectation: a difference map with no structure splits in exactly
    # these proportions. Without them "30% of the energy is high frequency" is not
    # readable as high or low, and `_over_uniform` is that comparison done once here
    # rather than by every reader.
    for name, _, _ in BANDS:
        out[f"spectrum_{name}_area"] = float(masks[name].mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(b - a))) ** 2
    total = float(power.sum())
    if total <= 0:
        # An identical pair has no difference energy to apportion. Zero fractions
        # would read as "all the energy is in the low band"; nan says "no effect".
        return out
    for name, _, _ in BANDS:
        fraction = float(power[masks[name]].sum() / total)
        area = out[f"spectrum_{name}_area"]
        out[f"spectrum_{name}_fraction"] = fraction
        out[f"spectrum_{name}_over_uniform"] = float(fraction / area) if area > 0 else float("nan")
    low = out["spectrum_low_fraction"]
    out["spectrum_high_over_low"] = (float(out["spectrum_high_fraction"] / low)
                                     if low > 0 else float("inf"))
    return out


def gradient_change(clean, treated) -> Dict[str, float]:
    """Change in Sobel magnitude: did the intervention sharpen or soften the image?

    ``gradient_energy_ratio`` below 1 means the treated image carries less edge
    energy than the clean one -- a loss of definition rather than a different scene.
    """
    a, b = to_gray(clean), to_gray(treated)
    out = {"gradient_rmse": float("nan"), "gradient_energy_ratio": float("nan"),
           "gradient_mean_clean": float("nan"), "gradient_mean_treated": float("nan")}
    if a is None or b is None or a.shape != b.shape:
        return out
    ga, gb = sobel_magnitude(a), sobel_magnitude(b)
    out["gradient_rmse"] = float(np.sqrt(np.mean((gb - ga) ** 2)))
    out["gradient_mean_clean"] = float(ga.mean())
    out["gradient_mean_treated"] = float(gb.mean())
    energy = float((ga ** 2).sum())
    if energy > 0:
        out["gradient_energy_ratio"] = float((gb ** 2).sum() / energy)
    return out


def high_gradient_concentration(clean, treated, quantile: float = 0.8) -> Dict[str, float]:
    """What share of the difference lands on the clean image's own structure.

    The mask is the top ``1 - quantile`` of clean Sobel magnitude -- contours, edges and
    highlights -- and is taken from the **clean** image alone, so it cannot be moved by
    the intervention being measured.

    ``concentration_ratio`` is the share of difference energy inside that mask divided
    by the mask's area fraction, so 1.0 means the effect is spread uniformly over the
    frame and values above 1 mean it is concentrated on structure.  Reporting the ratio
    rather than the bare share is what makes the number independent of the quantile.
    """
    a, b = to_gray(clean), to_gray(treated)
    out = {"detail_mask_fraction": float("nan"), "detail_energy_share": float("nan"),
           "concentration_ratio": float("nan"), "detail_quantile": float(quantile)}
    if a is None or b is None or a.shape != b.shape:
        return out
    gradient = sobel_magnitude(a)
    # Rank-based rather than value-based. A generated image often has large flat
    # regions, where `gradient >= quantile(gradient, q)` has its threshold at zero and
    # the mask silently becomes the whole frame -- which pins the ratio below to exactly
    # 1.0 and reads as "no concentration" when the truth is "the mask was undefined".
    # Taking the top (1 - q) of pixels by rank keeps the area equal to 1 - q whatever
    # the distribution of ties.
    flat = gradient.reshape(-1)
    keep = int(round((1.0 - min(max(quantile, 0.0), 1.0)) * flat.size))
    keep = min(max(keep, 1), flat.size)
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argsort(flat)[::-1][:keep]] = True
    mask = mask.reshape(gradient.shape)
    area = float(mask.mean())
    out["detail_mask_fraction"] = area
    difference = (b - a) ** 2
    total = float(difference.sum())
    if total <= 0 or area <= 0:
        return out
    share = float(difference[mask].sum() / total)
    out["detail_energy_share"] = share
    out["concentration_ratio"] = float(share / area)
    return out


def detail_metrics(clean, treated, *, quantile: float = 0.8) -> Dict[str, float]:
    """All three readouts in one row, for a metrics table."""
    row: Dict[str, float] = {}
    row.update(spectrum_bands(clean, treated))
    row.update(gradient_change(clean, treated))
    row.update(high_gradient_concentration(clean, treated, quantile=quantile))
    return row


def amplified_difference(clean, treated, factor: float = 10.0) -> Optional[np.ndarray]:
    """A viewable difference map, and the caller must state the factor beside it.

    Returned as uint8 centred on mid grey, so zero difference is flat 128 and the sign
    of the change is visible.  An amplified difference shown without its factor is not
    interpretable, which is why this returns only the array and never a figure.
    """
    a, b = to_gray(clean), to_gray(treated)
    if a is None or b is None or a.shape != b.shape:
        return None
    scaled = 128.0 + float(factor) * (b - a)
    return np.clip(scaled, 0, 255).astype(np.uint8)
