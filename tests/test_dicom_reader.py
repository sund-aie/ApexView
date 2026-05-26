"""Tests for the read-only DICOM input adapter.

Synthetic and deterministic. Each test builds a minimal DICOM in memory using
pydicom, writes it to ``tmp_path``, and reads it back via :func:`load_dicom`
to confirm what we put in is what we get out. No ``.dcm`` files are committed.
"""

from __future__ import annotations

import math

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from apexview.engine.extension_stitch import (
    InsufficientOverlapError,
    StitchResult,
    stitch_extension,
)
from apexview.io.dicom_reader import (
    InvalidDicomError,
    RadiographImage,
    load_dicom,
)


def _build_dicom(
    tmp_path,
    pixels: np.ndarray,
    *,
    name: str = "test.dcm",
    pixel_spacing=None,
    imager_pixel_spacing=None,
) -> str:
    """Write a minimal single-frame grayscale DICOM with the given pixel
    array and optional spacing tags. Returns the file path as a str."""
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
    if imager_pixel_spacing is not None:
        ds.ImagerPixelSpacing = list(imager_pixel_spacing)

    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def test_round_trip_with_pixel_spacing(tmp_path):
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(48, 64), dtype=np.uint8)
    path = _build_dicom(tmp_path, pixels, pixel_spacing=[0.1, 0.1])

    img = load_dicom(path)

    assert isinstance(img, RadiographImage)
    np.testing.assert_array_equal(img.pixels_raw, pixels)
    assert img.pixel_spacing_mm == (0.1, 0.1)
    assert img.pixel_spacing_source == "PixelSpacing"
    assert img.pixels_u8.dtype == np.uint8
    assert img.pixels_u8.ndim == 2
    assert img.bit_depth == 8


def test_imager_pixel_spacing_takes_priority(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    path = _build_dicom(
        tmp_path, pixels,
        pixel_spacing=[0.2, 0.2],
        imager_pixel_spacing=[0.05, 0.05],
    )

    img = load_dicom(path)

    assert img.pixel_spacing_mm == (0.05, 0.05)
    assert img.pixel_spacing_source == "ImagerPixelSpacing"


def test_missing_spacing_is_reported_honestly(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    path = _build_dicom(tmp_path, pixels)

    img = load_dicom(path)

    assert img.pixel_spacing_mm is None
    assert img.pixel_spacing_source == "unavailable"


def test_16bit_to_8bit_per_image_rescale(tmp_path):
    pixels = np.zeros((24, 32), dtype=np.uint16)
    rng = np.random.default_rng(13)
    pixels[:] = rng.integers(1500, 4500, size=pixels.shape, dtype=np.uint16)
    pixels[0, 0] = 1000
    pixels[-1, -1] = 5000

    path = _build_dicom(tmp_path, pixels, name="hi16.dcm")
    img = load_dicom(path)

    assert img.bit_depth == 16
    assert img.pixels_raw.dtype == np.uint16
    np.testing.assert_array_equal(img.pixels_raw, pixels)

    assert img.pixels_u8.dtype == np.uint8
    assert img.pixels_u8.shape == pixels.shape

    assert img.pixels_u8[0, 0] == 0, "image min should rescale to 0"
    assert img.pixels_u8[-1, -1] == 255, "image max should rescale to 255"
    assert int(img.pixels_u8.min()) == 0
    assert int(img.pixels_u8.max()) == 255


def test_pixels_u8_is_valid_engine_input(tmp_path):
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 4096, size=(120, 160), dtype=np.uint16)
    path = _build_dicom(tmp_path, pixels)

    img = load_dicom(path)

    try:
        result = stitch_extension(img.pixels_u8, img.pixels_u8)
    except InsufficientOverlapError:
        return
    except ValueError as exc:
        pytest.fail(f"pixels_u8 violated engine input contract: {exc}")
    assert isinstance(result, StitchResult)


def test_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.dcm"
    with pytest.raises(FileNotFoundError):
        load_dicom(missing)


def test_non_dicom_file_raises_invalid_dicom(tmp_path):
    bogus = tmp_path / "not_a.dcm"
    bogus.write_bytes(b"this is definitely not a DICOM file" * 8)
    with pytest.raises(InvalidDicomError):
        load_dicom(bogus)


def test_multi_dimensional_pixel_array_is_rejected(tmp_path):
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, size=(2, 16, 16), dtype=np.uint8)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = tmp_path / "multi.dcm"
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Multi"
    ds.PatientID = "0002"
    ds.Modality = "IO"
    ds.NumberOfFrames = 2
    ds.Rows, ds.Columns = frames.shape[1], frames.shape[2]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelData = frames.tobytes()
    ds.save_as(str(path), enforce_file_format=True)

    with pytest.raises(InvalidDicomError):
        load_dicom(path)
