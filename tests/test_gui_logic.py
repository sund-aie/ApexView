"""Headless tests for the GUI's presentation logic.

These tests must run without a display: they import the helpers from
``apexview.gui.app`` (``format_pixel_spacing``, ``format_analysis``,
``save_stitched_image``) and exercise them on synthetic
:class:`ClassificationResult` / :class:`RadiographImage` instances. No
QApplication is constructed, no Qt widget is instantiated.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Defensive: refuse the Qt platform entirely so any accidental Qt construction
# during import would surface fast instead of silently spawning a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apexview.engine.pair_classifier import ClassificationResult, PairType
from apexview.gui.app import (
    AnalysisDisplay,
    format_analysis,
    format_pixel_spacing,
    save_stitched_image,
)
from apexview.io.dicom_reader import RadiographImage


def _u8_image() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(24, 32), dtype=np.uint8)


def _radiograph(spacing=None, source: str = "unavailable") -> RadiographImage:
    arr = _u8_image()
    return RadiographImage(
        pixels_raw=arr,
        pixels_u8=arr,
        bit_depth=8,
        pixel_spacing_mm=spacing,
        pixel_spacing_source=source,
    )


# --------------------------------------------------------------------------
# format_pixel_spacing
# --------------------------------------------------------------------------
def test_format_pixel_spacing_known_value():
    img = _radiograph(spacing=(0.085, 0.085), source="ImagerPixelSpacing")
    text = format_pixel_spacing(img)
    assert "0.085" in text
    assert "ImagerPixelSpacing" in text


def test_format_pixel_spacing_unavailable_is_honest():
    img = _radiograph(spacing=None, source="unavailable")
    text = format_pixel_spacing(img)
    assert "unavailable" in text.lower()
    # No fabricated number snuck in.
    assert "mm x" not in text


# --------------------------------------------------------------------------
# format_analysis — EXTENSION
# --------------------------------------------------------------------------
def test_format_analysis_extension_enables_save_and_shows_image():
    stitched = np.full((40, 40), 200, dtype=np.uint8)
    result = ClassificationResult(
        pair_type=PairType.EXTENSION,
        inlier_count=248,
        mean_reprojection_error=0.189,
        stitched_image=stitched,
        message="Extension pair: 248 inliers (>= 100 threshold); stitched.",
    )

    display = format_analysis(result)

    assert isinstance(display, AnalysisDisplay)
    assert display.verdict_label == "EXTENSION"
    assert display.inlier_text == "Inliers: 248"
    assert "0.189" in display.reproj_text
    assert "px" in display.reproj_text
    assert display.message_text == result.message
    assert display.stitched_image is stitched
    assert display.has_stitched_image is True
    assert display.save_enabled is True


# --------------------------------------------------------------------------
# format_analysis — ANGULATION
# --------------------------------------------------------------------------
def test_format_analysis_angulation_disables_save_and_shows_message():
    result = ClassificationResult(
        pair_type=PairType.ANGULATION,
        inlier_count=34,
        mean_reprojection_error=1.230,
        stitched_image=None,
        message="Angulation detected: correction not yet implemented.",
    )

    display = format_analysis(result)

    assert display.verdict_label == "ANGULATION"
    assert display.inlier_text == "Inliers: 34"
    assert "1.230" in display.reproj_text
    assert display.message_text == result.message
    assert display.stitched_image is None
    assert display.has_stitched_image is False
    assert display.save_enabled is False


def test_format_analysis_angulation_with_stitched_image_still_hides_it():
    # Defensive: if a future engine ever returned a stitched_image alongside
    # an ANGULATION verdict (it currently does not), the GUI must still NOT
    # offer to save it — verdict drives the surface, not the field.
    stitched = np.zeros((10, 10), dtype=np.uint8)
    result = ClassificationResult(
        pair_type=PairType.ANGULATION,
        inlier_count=12,
        mean_reprojection_error=2.0,
        stitched_image=stitched,
        message="angulation",
    )
    display = format_analysis(result)
    assert display.has_stitched_image is False
    assert display.save_enabled is False
    assert display.stitched_image is None


# --------------------------------------------------------------------------
# NaN reprojection error renders "n/a"
# --------------------------------------------------------------------------
def test_format_analysis_nan_reproj_renders_n_a():
    result = ClassificationResult(
        pair_type=PairType.ANGULATION,
        inlier_count=0,
        mean_reprojection_error=math.nan,
        stitched_image=None,
        message="matching failed",
    )
    display = format_analysis(result)
    assert "n/a" in display.reproj_text
    assert "nan" not in display.reproj_text.lower().replace("n/a", "")


# --------------------------------------------------------------------------
# save_stitched_image — round-trip without Qt
# --------------------------------------------------------------------------
def test_save_stitched_image_writes_png(tmp_path):
    arr = np.full((16, 24), 180, dtype=np.uint8)
    out = tmp_path / "stitched.png"
    save_stitched_image(arr, out)
    assert out.exists() and out.stat().st_size > 0


def test_save_stitched_image_writes_tiff(tmp_path):
    arr = np.full((16, 24), 90, dtype=np.uint8)
    out = tmp_path / "stitched.tif"
    save_stitched_image(arr, out)
    assert out.exists() and out.stat().st_size > 0


def test_save_stitched_image_writes_jpeg(tmp_path):
    arr = np.full((16, 24), 50, dtype=np.uint8)
    out = tmp_path / "stitched.jpg"
    save_stitched_image(arr, out)
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------
# Module imports cleanly without instantiating Qt (headless guarantee)
# --------------------------------------------------------------------------
def test_module_import_does_not_construct_qapplication():
    # If importing apexview.gui.app constructed a QApplication, this would
    # have happened at import time above and a real QGuiApplication instance
    # would exist. We verify no Qt application object is alive.
    from apexview.gui import app as gui
    # The helpers and dataclass should be importable without PyQt6 even being
    # required, but PyQt6 is installed in this dev env. We assert that the
    # module did not eagerly build a QApplication.
    try:
        from PyQt6.QtWidgets import QApplication
        assert QApplication.instance() is None
    except ImportError:
        # PyQt6 unavailable -> the import of the gui module still succeeded
        # because Qt is imported lazily.
        pass
    # And the helpers we depend on are present.
    assert callable(gui.format_analysis)
    assert callable(gui.format_pixel_spacing)
    assert callable(gui.save_stitched_image)
