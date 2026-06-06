"""Tests for the DICOM sensor/device metadata reader.

Synthetic and deterministic. Each DICOM-shaped fixture is built in memory
with pydicom and written to ``tmp_path``; no real .dcm files are checked
into the repo.
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

from apexview.io.sensor_info import SensorInfo, read_sensor_info


def _build_dicom(
    tmp_path,
    pixels: np.ndarray,
    *,
    name: str = "test.dcm",
    manufacturer: str | None = None,
    model: str | None = None,
    serial: str | None = None,
    detector_type: str | None = None,
    imager_pixel_spacing=None,
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

    if manufacturer is not None:
        ds.Manufacturer = manufacturer
    if model is not None:
        ds.ManufacturerModelName = model
    if serial is not None:
        ds.DeviceSerialNumber = serial
    if detector_type is not None:
        ds.DetectorType = detector_type
    if imager_pixel_spacing is not None:
        ds.ImagerPixelSpacing = list(imager_pixel_spacing)

    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def test_dicom_with_all_sensor_tags_is_read(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    path = _build_dicom(
        tmp_path,
        pixels,
        manufacturer="Acme Imaging",
        model="ToothCam 9000",
        serial="SN-12345",
        detector_type="CMOS",
        imager_pixel_spacing=[0.085, 0.085],
    )

    info = read_sensor_info(path)

    assert isinstance(info, SensorInfo)
    assert info.manufacturer == "Acme Imaging"
    assert info.model == "ToothCam 9000"
    assert info.device_serial == "SN-12345"
    assert info.detector_type == "CMOS"
    assert info.imager_pixel_spacing_mm == (0.085, 0.085)

    assert "Acme Imaging" in info.summary
    assert "ToothCam 9000" in info.summary
    assert "SN-12345" in info.summary
    assert "CMOS" in info.summary
    assert "0.085" in info.summary
    assert "mm" in info.summary


def test_dicom_without_sensor_tags_reports_nothing_honestly(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    path = _build_dicom(tmp_path, pixels)

    info = read_sensor_info(path)

    assert info.manufacturer is None
    assert info.model is None
    assert info.device_serial is None
    assert info.detector_type is None
    assert info.imager_pixel_spacing_mm is None
    assert "no manufacturer/device tags" in info.summary.lower() or (
        "no sensor metadata" in info.summary.lower()
    )


def test_dicom_with_only_some_tags_omits_missing_parts(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    path = _build_dicom(
        tmp_path,
        pixels,
        manufacturer="Acme Imaging",
        imager_pixel_spacing=[0.1, 0.1],
    )

    info = read_sensor_info(path)

    assert info.manufacturer == "Acme Imaging"
    assert info.model is None
    assert info.device_serial is None
    assert info.detector_type is None
    assert info.imager_pixel_spacing_mm == (0.1, 0.1)

    assert "Acme Imaging" in info.summary
    assert "s/n" not in info.summary
    assert "detector " not in info.summary
    assert "0.100" in info.summary


def test_jpeg_path_returns_all_none_with_honest_summary(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    jpeg_path = tmp_path / "rad.jpg"
    Image.fromarray(pixels).convert("L").save(str(jpeg_path))

    info = read_sensor_info(jpeg_path)

    assert info.manufacturer is None
    assert info.model is None
    assert info.device_serial is None
    assert info.detector_type is None
    assert info.imager_pixel_spacing_mm is None
    assert "not a DICOM" in info.summary


def test_png_path_returns_all_none_with_honest_summary(tmp_path):
    pixels = np.zeros((16, 16), dtype=np.uint8)
    png_path = tmp_path / "rad.png"
    Image.fromarray(pixels).convert("L").save(str(png_path))

    info = read_sensor_info(png_path)

    assert info.manufacturer is None
    assert info.imager_pixel_spacing_mm is None
    assert "not a DICOM" in info.summary


def test_garbage_file_does_not_raise(tmp_path):
    bogus = tmp_path / "garbage.dcm"
    bogus.write_bytes(b"not a real dicom at all" * 16)

    info = read_sensor_info(bogus)

    assert info.manufacturer is None
    assert "not a DICOM" in info.summary


def test_missing_path_raises_file_not_found(tmp_path):
    missing = tmp_path / "nope.dcm"
    with pytest.raises(FileNotFoundError):
        read_sensor_info(missing)


def test_sensor_info_is_frozen():
    info = SensorInfo(
        manufacturer=None,
        model=None,
        device_serial=None,
        detector_type=None,
        imager_pixel_spacing_mm=None,
        summary="x",
    )
    with pytest.raises(Exception):
        info.manufacturer = "x"  # type: ignore[misc]
