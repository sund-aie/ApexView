"""Read-only DICOM input adapter for ApexView.

This module sits IN FRONT of the engine. It loads a DICOM file from disk and
returns a :class:`RadiographImage` carrying both the original pixel array (at
its native bit depth) and a uint8 2D grayscale version suitable as input to
:func:`apexview.engine.extension_stitch.stitch_extension`. It does not write,
export, or otherwise modify DICOM data.

Single source of truth (two parts):

1. **8-bit conversion.** Every path from a DICOM file to the SIFT engine
   goes through :func:`_to_uint8` here. The conversion rescales each image's
   actual ``[min, max]`` intensity range linearly to ``0..255`` (NOT a fixed
   16-bit max — real intraoral sensors rarely fill the full range, and using
   a fixed max would crush contrast on a typical dental film). Downstream
   code must read ``RadiographImage.pixels_u8``; it must NOT re-derive its
   own 8-bit version with different math.

2. **Pixel spacing.** Intraoral dental radiographs frequently lack true
   millimetre-per-pixel calibration because a flat sensor cannot know
   object-to-sensor distance. We read the spacing from DICOM if it is
   present, with ``ImagerPixelSpacing`` (0018,1164) preferred over
   ``PixelSpacing`` (0028,0030). If neither tag exists we report
   ``pixel_spacing_mm = None`` and ``pixel_spacing_source = "unavailable"``.
   We DO NOT substitute a hardcoded default such as ``0.05`` mm/px — that
   was the dishonest behavior of an earlier prototype and is exactly what
   this field must not do. Any measurement or UI code must surface
   "scale unknown" when this is None rather than fabricate millimetres.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError as _PydicomInvalidDicomError


class InvalidDicomError(Exception):
    """Raised when a file cannot be parsed as DICOM, or its content falls
    outside the supported subset (single-frame grayscale)."""


@dataclass
class RadiographImage:
    pixels_raw: np.ndarray
    pixels_u8: np.ndarray
    bit_depth: int
    pixel_spacing_mm: tuple[float, float] | None
    pixel_spacing_source: str


def _to_uint8(pixels: np.ndarray) -> np.ndarray:
    """Rescale the per-image ``[min, max]`` intensity range linearly to
    ``0..255``. 8-bit input is passed through unchanged.

    Per-image min/max (rather than a fixed bit-depth max) is intentional:
    real sensors rarely span the full dynamic range, and a fixed-max rescale
    would compress all the useful contrast into a narrow band of the 8-bit
    output. This is the single point where the conversion happens.
    """
    if pixels.dtype == np.uint8:
        return pixels
    lo = int(pixels.min())
    hi = int(pixels.max())
    if hi == lo:
        return np.zeros(pixels.shape, dtype=np.uint8)
    scaled = (pixels.astype(np.float64) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _read_pixel_spacing(
    dataset: pydicom.Dataset,
) -> tuple[tuple[float, float] | None, str]:
    """Return ``(spacing, source)``. Spacing is ``(row, col)`` in mm or
    ``None`` if absent. ``source`` names the DICOM tag we read, or
    ``"unavailable"``. Order: ImagerPixelSpacing then PixelSpacing.
    """
    for attr, source in (
        ("ImagerPixelSpacing", "ImagerPixelSpacing"),
        ("PixelSpacing", "PixelSpacing"),
    ):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        try:
            row = float(value[0])
            col = float(value[1])
        except (TypeError, IndexError, ValueError):
            continue
        return (row, col), source
    return None, "unavailable"


def _read_bit_depth(dataset: pydicom.Dataset, pixels: np.ndarray) -> int:
    bits_stored = getattr(dataset, "BitsStored", None)
    if bits_stored is not None:
        return int(bits_stored)
    return int(pixels.dtype.itemsize * 8)


def load_dicom(path: str | os.PathLike[str]) -> RadiographImage:
    """Load a single-frame grayscale DICOM and return a RadiographImage.

    Raises:
        FileNotFoundError: the path does not exist.
        InvalidDicomError: the file is not a DICOM, or is multi-frame /
            multi-channel (outside the supported subset for this task).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DICOM file not found: {p}")

    try:
        dataset = pydicom.dcmread(str(p))
        pixels_raw = dataset.pixel_array
    except _PydicomInvalidDicomError as exc:
        raise InvalidDicomError(f"Not a valid DICOM file: {p} ({exc})") from exc
    except Exception as exc:
        raise InvalidDicomError(f"Failed to read DICOM {p}: {exc}") from exc

    if pixels_raw.ndim != 2:
        raise InvalidDicomError(
            f"Only single-frame grayscale intraoral DICOMs are supported in "
            f"this task; got pixel array with shape {pixels_raw.shape}."
        )

    bit_depth = _read_bit_depth(dataset, pixels_raw)
    pixel_spacing_mm, pixel_spacing_source = _read_pixel_spacing(dataset)
    pixels_u8 = _to_uint8(pixels_raw)

    return RadiographImage(
        pixels_raw=pixels_raw,
        pixels_u8=pixels_u8,
        bit_depth=bit_depth,
        pixel_spacing_mm=pixel_spacing_mm,
        pixel_spacing_source=pixel_spacing_source,
    )
