"""Image input adapter for ApexView.

Loads a single image from disk and returns a
:class:`apexview.io.dicom_reader.RadiographImage` that the rest of the
engine already consumes. Dispatches by file extension:

* ``.dcm`` / ``.dicom`` / no extension -> :func:`load_dicom` (full DICOM
  metadata, including pixel spacing when present).
* ``.jpg`` / ``.jpeg`` / ``.png`` -> Pillow grayscale load. JPEG and PNG
  carry NO physical pixel-spacing metadata, so ``pixel_spacing_mm`` is set
  to ``None`` and the source string says so honestly. Real-world
  measurement from such images is not possible without an out-of-band scale
  reference; this is by design, not a bug.

Single source of truth: the engine's 8-bit conversion still lives in
:mod:`apexview.io.dicom_reader` for DICOM inputs. For JPEG/PNG, the image
is already 8-bit grayscale after the explicit ``"L"`` conversion below; the
same array is stored as both ``pixels_raw`` and ``pixels_u8`` because there
is no higher-bit-depth source to preserve.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from apexview.io.dicom_reader import RadiographImage, load_dicom

_PILLOW_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_DICOM_EXTENSIONS = {"", ".dcm", ".dicom"}


def _load_pillow_grayscale(path: Path) -> RadiographImage:
    """Load a JPEG/PNG as an 8-bit grayscale RadiographImage."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Pillow is required to load JPEG/PNG images") from exc

    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            array = np.array(gray, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not read image file {path}: {exc}") from exc

    if array.ndim != 2:
        raise ValueError(
            f"expected a 2D grayscale image after conversion, got shape {array.shape}"
        )

    return RadiographImage(
        pixels_raw=array,
        pixels_u8=array,
        bit_depth=8,
        pixel_spacing_mm=None,
        pixel_spacing_source="unavailable (JPEG/PNG carries no calibration)",
    )


def load_image(path: str | os.PathLike[str]) -> RadiographImage:
    """Load a DICOM, JPEG, or PNG image into a :class:`RadiographImage`.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is not a supported image, cannot be decoded,
            or (for DICOM) is not a single-frame grayscale study.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in _PILLOW_EXTENSIONS:
        return _load_pillow_grayscale(p)
    if suffix in _DICOM_EXTENSIONS:
        try:
            return load_dicom(str(p))
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read DICOM file {p}: {exc}") from exc
    raise ValueError(
        f"unsupported image extension {suffix!r}; expected one of "
        f"{sorted(_DICOM_EXTENSIONS | _PILLOW_EXTENSIONS)}"
    )
