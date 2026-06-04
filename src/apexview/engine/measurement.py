"""Detector-plane 2D distance measurement for ApexView.

WHAT THIS MODULE IS:
    A small, pure-library helper that converts a pixel distance between two
    image points into MILLIMETRES using a scale reference. Two construction
    paths for the scale are provided:

      * :meth:`ScaleCalibration.from_sensor_size` — anisotropic, computed
        per axis from the physical sensor dimensions and the image shape.
      * :meth:`ScaleCalibration.from_reference_length` — isotropic, derived
        from a single known length measured along an arbitrary direction in
        the image (one number cannot separate the two axes, so this path
        forces them equal).

WHAT THIS MODULE IS NOT:
    * NOT a 3D measurement. It does not use the A2 reconstruction at all.
    * NOT corrected for X-ray magnification. Anatomy sits in front of the
      detector, so projected size on the detector reads LARGER than true
      size, and the bias varies with depth. This module reports the size
      AS PROJECTED ON THE DETECTOR.
    * NOT corrected for beam angulation. An angulated pair will distort
      apparent in-plane length.

The number returned is therefore a bounded approximation of true anatomy
("size on the detector"), not a verified true length. The honesty flag
:attr:`Measurement.magnification_corrected` is always ``False`` to make this
hard to forget.

Coordinate convention (matches the rest of the engine: extension_stitch,
stereo_geometry): a 2D image point is ``(x, y)`` where ``x`` is the COLUMN
coordinate and ``y`` is the ROW coordinate. Numpy image shape is
``(rows, cols) == (height_px, width_px)``.

Single source of truth: every returned value (``distance_mm``,
``distance_px``, the calibration's per-axis scales) is computed once here
and stored. Consumers must read these fields and never recompute them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _is_finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


@dataclass(frozen=True)
class ScaleCalibration:
    mm_per_px_row: float
    mm_per_px_col: float
    source: str
    assumes_isotropic: bool

    @classmethod
    def from_sensor_size(
        cls,
        sensor_width_mm: float,
        sensor_height_mm: float,
        image_shape: tuple[int, int],
    ) -> "ScaleCalibration":
        """Build an anisotropic calibration from the sensor's physical size.

        Sensor width corresponds to image columns (the x axis); sensor
        height corresponds to image rows (the y axis). The two axes are
        computed independently and not forced equal.
        """
        if not isinstance(sensor_width_mm, (int, float)) or not _is_finite_positive(
            float(sensor_width_mm)
        ):
            raise ValueError(
                f"sensor_width_mm must be finite and > 0, got {sensor_width_mm!r}"
            )
        if not isinstance(sensor_height_mm, (int, float)) or not _is_finite_positive(
            float(sensor_height_mm)
        ):
            raise ValueError(
                f"sensor_height_mm must be finite and > 0, got {sensor_height_mm!r}"
            )
        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or not all(isinstance(d, int) and d > 0 for d in image_shape)
        ):
            raise ValueError(
                f"image_shape must be a (rows, cols) tuple of positive ints, "
                f"got {image_shape!r}"
            )
        rows, cols = image_shape
        return cls(
            mm_per_px_row=float(sensor_height_mm) / float(rows),
            mm_per_px_col=float(sensor_width_mm) / float(cols),
            source="sensor_physical_size",
            assumes_isotropic=False,
        )

    @classmethod
    def from_reference_length(
        cls, known_mm: float, measured_px: float
    ) -> "ScaleCalibration":
        """Build an isotropic calibration from one known length.

        A single reference length measured along an arbitrary image
        direction cannot separate the row and column scales, so this path
        ASSUMES SQUARE PIXELS and forces ``mm_per_px_row == mm_per_px_col``.
        For genuinely anisotropic sensors, prefer
        :meth:`from_sensor_size`.
        """
        if not isinstance(known_mm, (int, float)) or not _is_finite_positive(
            float(known_mm)
        ):
            raise ValueError(f"known_mm must be finite and > 0, got {known_mm!r}")
        if not isinstance(measured_px, (int, float)) or not _is_finite_positive(
            float(measured_px)
        ):
            raise ValueError(
                f"measured_px must be finite and > 0, got {measured_px!r}"
            )
        s = float(known_mm) / float(measured_px)
        return cls(
            mm_per_px_row=s,
            mm_per_px_col=s,
            source="known_reference_length",
            assumes_isotropic=True,
        )


@dataclass(frozen=True)
class Measurement:
    distance_mm: float
    distance_px: float
    calibration: ScaleCalibration
    magnification_corrected: bool


def _validate_point(name: str, point: Sequence[float]) -> tuple[float, float]:
    try:
        length = len(point)
    except TypeError as exc:
        raise ValueError(f"{name} must be a length-2 sequence (x, y)") from exc
    if length != 2:
        raise ValueError(f"{name} must be length-2 (x, y), got length {length}")
    x, y = point[0], point[1]
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError(f"{name} coordinates must be numeric, got {point!r}")
    fx, fy = float(x), float(y)
    if not (math.isfinite(fx) and math.isfinite(fy)):
        raise ValueError(f"{name} coordinates must be finite, got {point!r}")
    return fx, fy


def measure_distance(
    point_a: Sequence[float],
    point_b: Sequence[float],
    calibration: ScaleCalibration,
) -> Measurement:
    """Distance between two image points in pixels and in millimetres.

    Each point is ``(x, y)`` where x is the column index and y is the row
    index. Per-axis conversion is applied SEPARATELY before combining:
    horizontal pixel delta is converted using ``mm_per_px_col`` and vertical
    pixel delta using ``mm_per_px_row``. This is the only correct way to
    handle anisotropic pixel spacing; scaling a single scalar pixel
    distance by one factor would silently fold the two axes together.
    """
    if not isinstance(calibration, ScaleCalibration):
        raise ValueError(
            f"calibration must be a ScaleCalibration, got {type(calibration).__name__}"
        )
    if not _is_finite_positive(calibration.mm_per_px_row):
        raise ValueError(
            f"calibration.mm_per_px_row must be finite and > 0, got "
            f"{calibration.mm_per_px_row!r}"
        )
    if not _is_finite_positive(calibration.mm_per_px_col):
        raise ValueError(
            f"calibration.mm_per_px_col must be finite and > 0, got "
            f"{calibration.mm_per_px_col!r}"
        )
    xa, ya = _validate_point("point_a", point_a)
    xb, yb = _validate_point("point_b", point_b)

    dx = xb - xa
    dy = yb - ya
    distance_px = math.hypot(dx, dy)
    dx_mm = dx * calibration.mm_per_px_col
    dy_mm = dy * calibration.mm_per_px_row
    distance_mm = math.hypot(dx_mm, dy_mm)

    return Measurement(
        distance_mm=distance_mm,
        distance_px=distance_px,
        calibration=calibration,
        magnification_corrected=False,
    )
