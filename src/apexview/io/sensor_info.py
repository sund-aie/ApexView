"""Read-only sensor/device metadata extractor for ApexView.

A small companion to :mod:`apexview.io.dicom_reader`. It does NOT modify
or replace the existing reader. Given a file path it pulls the
manufacturer/device tags from a DICOM and returns a :class:`SensorInfo`
plus a one-line ``summary`` the GUI can drop beneath the pixel-spacing
line. The intent is to stop ignoring metadata the sensor already carries.

Honesty: every field that the DICOM does not contain stays ``None``; the
summary omits the missing parts rather than inventing them. JPEG/PNG paths
(or anything else not parseable as DICOM) return an all-``None``
:class:`SensorInfo` with a summary that says so plainly. The function only
raises :class:`FileNotFoundError` for missing paths; it never raises on a
non-DICOM input so a GUI never crashes on a JPEG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pydicom


@dataclass(frozen=True)
class SensorInfo:
    manufacturer: str | None
    model: str | None
    device_serial: str | None
    detector_type: str | None
    imager_pixel_spacing_mm: tuple[float, float] | None
    summary: str


_NO_DICOM_SUMMARY = "No sensor metadata (not a DICOM)."
_NO_TAGS_SUMMARY = "Sensor: no manufacturer/device tags present in DICOM."


def _clean_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _clean_imager_pixel_spacing(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        row = float(value[0])
        col = float(value[1])
    except (TypeError, IndexError, ValueError):
        return None
    return (row, col)


def _compose_summary(
    manufacturer: str | None,
    model: str | None,
    serial: str | None,
    detector: str | None,
    spacing: tuple[float, float] | None,
) -> str:
    parts: list[str] = []

    name_pieces = [p for p in (manufacturer, model) if p]
    if name_pieces:
        device = " ".join(name_pieces)
        if serial:
            parts.append(f"{device} (s/n {serial})")
        else:
            parts.append(device)
    elif serial:
        parts.append(f"s/n {serial}")

    if detector:
        parts.append(f"detector {detector}")

    if spacing is not None:
        row, col = spacing
        parts.append(f"imager spacing {row:.3f}x{col:.3f} mm")

    if not parts:
        return _NO_TAGS_SUMMARY
    return "Sensor: " + ", ".join(parts) + "."


def read_sensor_info(path: str | os.PathLike[str]) -> SensorInfo:
    """Return the sensor/device metadata carried by a DICOM file.

    Reads Manufacturer (0008,0070), ManufacturerModelName (0008,1090),
    DeviceSerialNumber (0018,1000), DetectorType (0018,7004), and
    ImagerPixelSpacing (0018,1164). Any missing tag becomes ``None`` for
    that field; ``summary`` is composed only from the fields that are
    actually present.

    JPEG/PNG, corrupted DICOMs, or other unreadable files return an
    all-``None`` :class:`SensorInfo` with the "not a DICOM" summary. Only
    a non-existent path raises :class:`FileNotFoundError`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image file not found: {p}")

    empty = SensorInfo(
        manufacturer=None,
        model=None,
        device_serial=None,
        detector_type=None,
        imager_pixel_spacing_mm=None,
        summary=_NO_DICOM_SUMMARY,
    )

    try:
        dataset = pydicom.dcmread(str(p), stop_before_pixels=True)
    except Exception:
        return empty

    manufacturer = _clean_string(getattr(dataset, "Manufacturer", None))
    model = _clean_string(getattr(dataset, "ManufacturerModelName", None))
    serial = _clean_string(getattr(dataset, "DeviceSerialNumber", None))
    detector = _clean_string(getattr(dataset, "DetectorType", None))
    spacing = _clean_imager_pixel_spacing(
        getattr(dataset, "ImagerPixelSpacing", None)
    )

    summary = _compose_summary(manufacturer, model, serial, detector, spacing)

    return SensorInfo(
        manufacturer=manufacturer,
        model=model,
        device_serial=serial,
        detector_type=detector,
        imager_pixel_spacing_mm=spacing,
        summary=summary,
    )
