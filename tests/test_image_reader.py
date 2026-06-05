"""Tests for the image input adapter (DICOM + JPEG/PNG).

Deterministic, synthetic. No real files are committed. PNGs / JPEGs are
written into ``tmp_path`` per test and the loader is exercised against
them; the DICOM path is exercised via the same minimal pydicom builder
used elsewhere in the suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from apexview.io.dicom_reader import RadiographImage
from apexview.io.image_reader import load_image


# --------------------------------------------------------------------------
# Synthetic DICOM builder (copied scaffold from test_dicom_reader.py)
# --------------------------------------------------------------------------
def _build_dicom(
    tmp_path,
    pixels: np.ndarray,
    *,
    name: str = "test.dcm",
    pixel_spacing=None,
) -> str:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = tmp_path / name
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Synthetic"
    ds.PatientID = "0001"
    ds.Modality = "IO"

    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    bits = pixels.dtype.itemsize * 8
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelData = pixels.tobytes()

    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)

    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


# --------------------------------------------------------------------------
# PNG / JPEG round-trip via Pillow
# --------------------------------------------------------------------------
def _gradient(h: int = 60, w: int = 80) -> np.ndarray:
    row = np.linspace(20, 230, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def test_png_loads_as_2d_uint8_with_no_calibration(tmp_path):
    arr = _gradient()
    path = tmp_path / "sample.png"
    Image.fromarray(arr).save(str(path))

    img = load_image(path)

    assert isinstance(img, RadiographImage)
    assert img.pixels_u8.dtype == np.uint8
    assert img.pixels_u8.ndim == 2
    assert img.pixels_u8.shape == arr.shape
    assert img.bit_depth == 8
    assert img.pixel_spacing_mm is None
    assert "unavailable" in img.pixel_spacing_source.lower()
    # raw and u8 are the same array for 8-bit sources.
    np.testing.assert_array_equal(img.pixels_raw, img.pixels_u8)


def test_jpeg_loads_as_2d_uint8_with_no_calibration(tmp_path):
    arr = _gradient()
    path = tmp_path / "sample.jpg"
    Image.fromarray(arr).save(str(path), quality=92)

    img = load_image(path)

    assert img.pixels_u8.dtype == np.uint8
    assert img.pixels_u8.ndim == 2
    assert img.pixels_u8.shape == arr.shape
    assert img.bit_depth == 8
    assert img.pixel_spacing_mm is None
    assert "unavailable" in img.pixel_spacing_source.lower()


def test_rgb_png_is_converted_to_grayscale(tmp_path):
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red dominant
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(str(path))

    img = load_image(path)

    assert img.pixels_u8.ndim == 2
    assert img.pixels_u8.shape == (20, 30)


# --------------------------------------------------------------------------
# DICOM path still routes correctly and preserves spacing
# --------------------------------------------------------------------------
def test_dicom_route_preserves_pixel_spacing(tmp_path):
    arr = (np.random.default_rng(0)
           .integers(0, 256, size=(40, 50), dtype=np.uint8))
    path = _build_dicom(tmp_path, arr, pixel_spacing=[0.085, 0.085])

    img = load_image(path)

    assert img.pixel_spacing_mm == (0.085, 0.085)
    assert img.pixel_spacing_source == "PixelSpacing"
    assert img.pixels_u8.dtype == np.uint8


def test_dicom_without_extension_still_dispatched_to_dicom(tmp_path):
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[0, 0] = 100
    path = _build_dicom(tmp_path, arr, name="study0001")

    img = load_image(path)

    assert isinstance(img, RadiographImage)
    assert img.pixel_spacing_mm is None
    assert img.pixel_spacing_source == "unavailable"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_bogus_png_raises_value_error(tmp_path):
    p = tmp_path / "junk.png"
    p.write_bytes(b"definitely not a real PNG" * 8)
    with pytest.raises(ValueError):
        load_image(p)


def test_bogus_dicom_raises_value_error(tmp_path):
    p = tmp_path / "junk.dcm"
    p.write_bytes(b"definitely not a DICOM" * 8)
    with pytest.raises(ValueError):
        load_image(p)


def test_unsupported_extension_raises_value_error(tmp_path):
    p = tmp_path / "weird.tiff"
    p.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError):
        load_image(p)
