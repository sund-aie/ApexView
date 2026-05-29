"""Tests for the ApexView CLI (thin client over reader + engine).

Synthetic and deterministic. DICOMs are built in ``tmp_path`` with pydicom
(builder copied from test_dicom_reader.py). The extension pair reuses the
in-plane warp from test_extension_stitch.py; the angulation pair reuses the
two-layer parallax scene from test_pair_classifier.py. The CLI is driven
through ``main(argv)`` with output captured via capsys.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from apexview.cli import main

IMG_H = 240
IMG_W = 320
FOCAL = float(IMG_W)
BG_DEPTH = 3.0 * FOCAL
FG_DEPTH = 1.5 * FOCAL
AXIS_DEPTH = 2.0 * FOCAL


# --------------------------------------------------------------------------
# Synthetic DICOM builder (copied scaffold from test_dicom_reader.py)
# --------------------------------------------------------------------------
def _build_dicom(
    tmp_path,
    pixels: np.ndarray,
    *,
    name: str = "test.dcm",
    pixel_spacing=None,
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

    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)
    if imager_pixel_spacing is not None:
        ds.ImagerPixelSpacing = list(imager_pixel_spacing)

    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


# --------------------------------------------------------------------------
# Image generators (copied scaffold from the engine tests)
# --------------------------------------------------------------------------
def _make_background(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)
    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 40
            else:
                img[y : y + tile, x : x + tile] = 215
    rng = np.random.default_rng(seed * 31 + 7)
    for _ in range(20):
        cx = int(rng.integers(5, w - 5))
        cy = int(rng.integers(5, h - 5))
        cv2.circle(img, (cx, cy), 2, int(rng.integers(0, 256)), -1)
    return img


def _make_foreground(
    seed: int, h: int = IMG_H, w: int = IMG_W
) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        radius = int(rng.integers(3, 8))
        color = int(rng.integers(0, 256))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    for _ in range(40):
        x1 = int(rng.integers(5, w - 25))
        y1 = int(rng.integers(5, h - 25))
        side = int(rng.integers(6, 18))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color, -1)
        cv2.rectangle(mask, (x1, y1), (x1 + side, y1 + side), 255, -1)
    return img, mask


def _make_single_layer_image(seed: int) -> np.ndarray:
    img = _make_background(seed)
    fg, mask = _make_foreground(seed)
    img[mask > 127] = fg[mask > 127]
    return img


def _extension_homography(angle_deg: float, dx: float, dy: float = 0.0) -> np.ndarray:
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


def _layer_homography(angle_deg: float, layer_z: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    src = np.float32([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]])
    dst = []
    for u, v in src:
        X = (u - cx) * layer_z / FOCAL
        Y = (v - cy) * layer_z / FOCAL
        Zr = layer_z - AXIS_DEPTH
        Xn = X * c + Zr * s
        Zn = -X * s + Zr * c
        Zf = Zn + AXIS_DEPTH
        if Zf <= 1e-3:
            return np.eye(3, dtype=np.float64)
        dst.append([cx + FOCAL * Xn / Zf, cy + FOCAL * Y / Zf])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _render_two_layer_scene(seed: int, angle_deg: float) -> np.ndarray:
    bg = _make_background(seed)
    fg, mask = _make_foreground(seed)
    h_bg = _layer_homography(angle_deg, BG_DEPTH)
    h_fg = _layer_homography(angle_deg, FG_DEPTH)
    warped_bg = cv2.warpPerspective(bg, h_bg, (IMG_W, IMG_H))
    warped_fg = cv2.warpPerspective(fg, h_fg, (IMG_W, IMG_H))
    warped_mask = cv2.warpPerspective(mask, h_fg, (IMG_W, IMG_H))
    out = warped_bg.copy()
    out[warped_mask > 127] = warped_fg[warped_mask > 127]
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_inspect_reports_spacing_when_present(tmp_path, capsys):
    pixels = _make_single_layer_image(11)
    path = _build_dicom(tmp_path, pixels, pixel_spacing=[0.1, 0.1])

    code = main(["inspect", path])
    out = capsys.readouterr().out

    assert code == 0
    assert "0.100 mm x 0.100 mm" in out
    assert "PixelSpacing" in out


def test_inspect_reports_unavailable_honestly(tmp_path, capsys):
    pixels = _make_single_layer_image(11)
    path = _build_dicom(tmp_path, pixels)

    code = main(["inspect", path])
    out = capsys.readouterr().out

    assert code == 0
    assert "UNAVAILABLE" in out
    assert "0.05" not in out


def test_analyze_extension_saves_output(tmp_path, capsys):
    image_a = _make_single_layer_image(11)
    image_b = cv2.warpPerspective(
        image_a, _extension_homography(1.5, 60.0), (IMG_W, IMG_H)
    )
    path_a = _build_dicom(tmp_path, image_a, name="a.dcm", pixel_spacing=[0.1, 0.1])
    path_b = _build_dicom(tmp_path, image_b, name="b.dcm", pixel_spacing=[0.1, 0.1])
    out_png = tmp_path / "stitched.png"

    code = main(["analyze", path_a, path_b, "--out", str(out_png)])
    out = capsys.readouterr().out

    assert code == 0
    assert "EXTENSION" in out
    assert "Inlier count:" in out
    assert out_png.exists(), "stitched PNG was not written to disk"
    assert out_png.stat().st_size > 0


def test_analyze_angulation_writes_no_image(tmp_path, capsys):
    image_a = _render_two_layer_scene(11, 0.0)
    image_b = _render_two_layer_scene(11, 18.0)
    path_a = _build_dicom(tmp_path, image_a, name="a.dcm")
    path_b = _build_dicom(tmp_path, image_b, name="b.dcm")
    out_png = tmp_path / "should_not_exist.png"

    code = main(["analyze", path_a, path_b, "--out", str(out_png)])
    out = capsys.readouterr().out

    assert code == 0
    assert "ANGULATION" in out
    assert not out_png.exists(), "no image should be written for angulation"


def test_extension_and_angulation_give_different_verdicts(tmp_path, capsys):
    # Extension pair
    a = _make_single_layer_image(11)
    b = cv2.warpPerspective(a, _extension_homography(1.5, 60.0), (IMG_W, IMG_H))
    pa = _build_dicom(tmp_path, a, name="ea.dcm")
    pb = _build_dicom(tmp_path, b, name="eb.dcm")
    main(["analyze", pa, pb])
    ext_out = capsys.readouterr().out

    # Angulation pair
    c = _render_two_layer_scene(11, 0.0)
    d = _render_two_layer_scene(11, 18.0)
    pc = _build_dicom(tmp_path, c, name="ca.dcm")
    pd = _build_dicom(tmp_path, d, name="cb.dcm")
    main(["analyze", pc, pd])
    ang_out = capsys.readouterr().out

    assert "EXTENSION" in ext_out
    assert "ANGULATION" in ang_out


def test_file_not_found_is_clean_error(tmp_path, capsys):
    missing = str(tmp_path / "nope.dcm")
    code = main(["inspect", missing])
    captured = capsys.readouterr()

    assert code == 2
    assert "not found" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_not_a_dicom_is_clean_error(tmp_path, capsys):
    bogus = tmp_path / "bogus.dcm"
    bogus.write_bytes(b"definitely not a dicom" * 16)

    code = main(["inspect", str(bogus)])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err.strip() != ""
    assert "Traceback" not in captured.err


def test_no_args_shows_help_and_nonzero(capsys):
    code = main([])
    captured = capsys.readouterr()

    assert code != 0
    combined = captured.out + captured.err
    assert "usage" in combined.lower()
