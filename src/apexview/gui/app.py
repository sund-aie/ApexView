"""Minimal PyQt6 desktop GUI for ApexView.

THIN CLIENT. The GUI performs NO image analysis. It calls the existing
engine + reader and only displays what they already produced:

  * :func:`apexview.io.image_reader.load_image` for DICOM / JPEG / PNG
    input, returning a :class:`RadiographImage`.
  * :func:`apexview.engine.pair_classifier.classify_pair` for the verdict,
    inlier count, mean reprojection error, engine message, and (for
    extension pairs) the engine's already-stitched output image.

Every value shown in the UI is read verbatim from those objects. The GUI
deliberately does not import cv2: PNG/TIFF/JPEG export of the engine's
stitched array goes through Pillow.

Structure: presentation logic (string formatting, button-enable rules) is
factored into pure functions and dataclasses below so the unit tests can
exercise them HEADLESS, without constructing a QApplication. The Qt window
itself is built only inside :func:`main` and the class
:class:`ApexViewWindow`, which is the actual application entry point.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from apexview.engine.pair_classifier import (
    ClassificationResult,
    PairType,
    classify_pair,
)
from apexview.io.dicom_reader import RadiographImage
from apexview.io.image_reader import load_image


# -- Pure-logic helpers (no Qt; tested headlessly) -----------------------------


@dataclass(frozen=True)
class AnalysisDisplay:
    """Strings + flags that drive the UI after an Analyze step.

    Computed once from a :class:`ClassificationResult`; nothing in the GUI
    recomputes engine-owned values."""

    verdict_label: str
    inlier_text: str
    reproj_text: str
    message_text: str
    stitched_image: np.ndarray | None
    save_enabled: bool
    has_stitched_image: bool


def format_pixel_spacing(image: RadiographImage) -> str:
    """Render a RadiographImage's spacing status as one human-readable line."""
    if image.pixel_spacing_mm is None:
        return f"Pixel spacing: {image.pixel_spacing_source}"
    row_mm, col_mm = image.pixel_spacing_mm
    return (
        f"Pixel spacing: {row_mm:.3f} mm x {col_mm:.3f} mm "
        f"(from {image.pixel_spacing_source})"
    )


def format_analysis(result: ClassificationResult) -> AnalysisDisplay:
    """Build the GUI display state from a ClassificationResult."""
    is_extension = result.pair_type is PairType.EXTENSION
    has_stitched = is_extension and result.stitched_image is not None
    if math.isnan(result.mean_reprojection_error):
        reproj = "n/a"
    else:
        reproj = f"{result.mean_reprojection_error:.3f} px"
    return AnalysisDisplay(
        verdict_label=result.pair_type.name,
        inlier_text=f"Inliers: {result.inlier_count}",
        reproj_text=f"Mean reprojection error: {reproj}",
        message_text=result.message,
        stitched_image=result.stitched_image if has_stitched else None,
        save_enabled=has_stitched,
        has_stitched_image=has_stitched,
    )


def save_stitched_image(array: np.ndarray, path: str | Path) -> None:
    """Write an engine-produced uint8 grayscale array to disk via Pillow.

    Pillow handles the file-format encoding (the extension drives PNG /
    TIFF / JPEG selection); the array is produced by the engine.
    """
    from PIL import Image

    Image.fromarray(array).save(str(path))


# -- File-dialog filter and extension helpers --------------------------------

SUPPORTED_FILE_FILTER = (
    "Radiograph images (*.dcm *.dicom *.png *.jpg *.jpeg);;All files (*)"
)
SAVE_FILE_FILTER = "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)"


# -- Qt window ---------------------------------------------------------------

# Qt is imported lazily inside the classes/functions that use it, so that
# importing this module (for the pure helpers) does not require a display
# or even PyQt6 being installed at test time.


def _qt():
    """Late-import PyQt6 components. Raised ImportError surfaces clearly."""
    from PyQt6 import QtCore, QtGui, QtWidgets

    return QtCore, QtGui, QtWidgets


def _array_to_qpixmap(array: np.ndarray, max_size: int = 320):
    """Wrap a uint8 2D grayscale numpy array as a QPixmap, scaled to fit."""
    QtCore, QtGui, _ = _qt()
    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError("preview requires a 2D uint8 array")
    h, w = array.shape
    # Ensure a contiguous buffer so QImage's view is valid for its lifetime.
    contiguous = np.ascontiguousarray(array)
    qimg = QtGui.QImage(
        contiguous.data, w, h, contiguous.strides[0],
        QtGui.QImage.Format.Format_Grayscale8,
    ).copy()  # copy detaches from the numpy buffer
    pix = QtGui.QPixmap.fromImage(qimg)
    return pix.scaled(
        max_size, max_size,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )


class ApexViewWindow:
    """The single application window. Built lazily so import is headless-safe."""

    def __init__(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._QtWidgets = QtWidgets

        self.image_a: RadiographImage | None = None
        self.image_b: RadiographImage | None = None
        self.path_a: Path | None = None
        self.path_b: Path | None = None
        self.last_display: AnalysisDisplay | None = None

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("ApexView")
        self.window.resize(900, 700)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Image-pair panel
        pair_row = QtWidgets.QHBoxLayout()
        root.addLayout(pair_row)
        self._slot_a = self._build_slot("A", self._on_load_a)
        self._slot_b = self._build_slot("B", self._on_load_b)
        pair_row.addLayout(self._slot_a["layout"])
        pair_row.addLayout(self._slot_b["layout"])

        # Analyze button
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._on_analyze)
        root.addWidget(self.analyze_button)

        # Results panel
        self.verdict_label = QtWidgets.QLabel("")
        verdict_font = self.verdict_label.font()
        verdict_font.setPointSize(verdict_font.pointSize() + 6)
        verdict_font.setBold(True)
        self.verdict_label.setFont(verdict_font)
        self.inlier_label = QtWidgets.QLabel("")
        self.reproj_label = QtWidgets.QLabel("")
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)
        for w in (self.verdict_label, self.inlier_label, self.reproj_label, self.message_label):
            root.addWidget(w)

        # Stitched-image preview + save
        self.stitched_label = QtWidgets.QLabel("")
        self.stitched_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stitched_label.setMinimumHeight(200)
        root.addWidget(self.stitched_label, stretch=1)

        self.save_button = QtWidgets.QPushButton("Save stitched radiograph...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save)
        root.addWidget(self.save_button)

        # Status line
        self.status_label = QtWidgets.QLabel("Ready.")
        root.addWidget(self.status_label)

    # -- slot factory --------------------------------------------------------
    def _build_slot(self, label: str, on_click: Callable[[], None]) -> dict:
        QtCore, _QtGui, QtWidgets = self._QtCore, self._QtGui, self._QtWidgets
        layout = QtWidgets.QVBoxLayout()
        button = QtWidgets.QPushButton(f"Load Image {label}")
        button.clicked.connect(on_click)
        layout.addWidget(button)
        name = QtWidgets.QLabel("(no file)")
        name.setWordWrap(True)
        layout.addWidget(name)
        spacing = QtWidgets.QLabel("")
        spacing.setWordWrap(True)
        layout.addWidget(spacing)
        preview = QtWidgets.QLabel("")
        preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumSize(320, 240)
        layout.addWidget(preview)
        return {
            "layout": layout, "button": button,
            "name": name, "spacing": spacing, "preview": preview,
        }

    # -- handlers ------------------------------------------------------------
    def _on_load_a(self) -> None:
        self._load_into("A")

    def _on_load_b(self) -> None:
        self._load_into("B")

    def _load_into(self, slot: str) -> None:
        QtWidgets = self._QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, f"Select image {slot}", "", SUPPORTED_FILE_FILTER,
        )
        if not path:
            return
        try:
            image = load_image(path)
        except (FileNotFoundError, ValueError) as exc:
            self._show_error(f"Could not load image {slot}: {exc}")
            return
        ui = self._slot_a if slot == "A" else self._slot_b
        ui["name"].setText(Path(path).name)
        ui["spacing"].setText(format_pixel_spacing(image))
        try:
            ui["preview"].setPixmap(_array_to_qpixmap(image.pixels_u8))
        except Exception as exc:
            self._show_error(f"Could not render preview for {slot}: {exc}")
        if slot == "A":
            self.image_a = image
            self.path_a = Path(path)
        else:
            self.image_b = image
            self.path_b = Path(path)
        self.analyze_button.setEnabled(
            self.image_a is not None and self.image_b is not None
        )
        self.status_label.setText(f"Loaded image {slot}.")

    def _on_analyze(self) -> None:
        if self.image_a is None or self.image_b is None:
            return
        try:
            result = classify_pair(self.image_a.pixels_u8, self.image_b.pixels_u8)
        except ValueError as exc:
            self._show_error(f"Engine rejected the inputs: {exc}")
            return
        except Exception as exc:  # surface unexpected engine failures cleanly
            self._show_error(f"Unexpected engine error: {exc}")
            return

        display = format_analysis(result)
        self.last_display = display
        self.verdict_label.setText(display.verdict_label)
        self.inlier_label.setText(display.inlier_text)
        self.reproj_label.setText(display.reproj_text)
        self.message_label.setText(display.message_text)
        if display.has_stitched_image and display.stitched_image is not None:
            try:
                self.stitched_label.setPixmap(
                    _array_to_qpixmap(display.stitched_image, max_size=720)
                )
            except Exception as exc:
                self._show_error(f"Could not render stitched preview: {exc}")
                self.stitched_label.clear()
        else:
            self.stitched_label.clear()
        self.save_button.setEnabled(display.save_enabled)
        self.status_label.setText("Analysis complete.")

    def _on_save(self) -> None:
        QtWidgets = self._QtWidgets
        if self.last_display is None or self.last_display.stitched_image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Save stitched radiograph", "stitched.png", SAVE_FILE_FILTER,
        )
        if not path:
            return
        try:
            save_stitched_image(self.last_display.stitched_image, path)
        except Exception as exc:
            self._show_error(f"Could not save image: {exc}")
            return
        self.status_label.setText(f"Saved stitched image to {path}.")

    def _show_error(self, message: str) -> None:
        QtWidgets = self._QtWidgets
        self.status_label.setText(message)
        QtWidgets.QMessageBox.warning(self.window, "ApexView", message)

    def show(self) -> None:
        self.window.show()


def main() -> int:
    """Console entry point. Creates the QApplication and shows the window."""
    QtCore, _QtGui, QtWidgets = _qt()
    app = QtWidgets.QApplication(sys.argv)
    window = ApexViewWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
