"""Minimal PyQt6 desktop GUI for ApexView.

THIN CLIENT. The GUI performs NO image analysis. It calls the existing
engine + reader and only displays what they already produced:

  * :func:`apexview.io.image_reader.load_image` for DICOM / JPEG / PNG
    input, returning a :class:`RadiographImage`.
  * :func:`apexview.io.sensor_info.read_sensor_info` for the
    manufacturer/device summary surfaced beneath each loaded image (JPEG
    or PNG honestly reports "not a DICOM").
  * :func:`apexview.engine.pair_classifier.classify_pair` for the verdict,
    inlier count, mean reprojection error, engine message, and (for
    extension pairs) the engine's already-stitched output image.
  * :func:`apexview.engine.pivot_align.align_for_pivot` for angulation
    pairs: a rough overlap registration of two real different-angle
    radiographs so a single pivot viewer can flip between them. This is
    NOT a stitch, NOT an angulation correction, NOT a 3D reconstruction.

Every value shown in the UI is read verbatim from those objects. The GUI
deliberately does not import cv2: PNG/TIFF/JPEG export of arrays goes
through Pillow, and the pivot viewer's blend between image_a and
warped_b is pure numpy arithmetic on engine-produced arrays (display
math only; no analysis).

Layout: the whole window is hosted inside a :class:`QScrollArea` so the
content is scrollable when it exceeds the viewport, and image displays
scale to the available panel size while preserving aspect ratio. Each
image label keeps its full-resolution :class:`QPixmap` and re-fits on
resize, so the previews and the pivot frame stay legible at any window
size.

Structure: presentation logic (string formatting, button-enable rules,
the pivot blend) is factored into pure functions and dataclasses below so
the unit tests can exercise them HEADLESS, without constructing a
QApplication. The Qt window itself is built only inside :func:`main` and
the class :class:`ApexViewWindow`, which is the actual application entry
point.
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
from apexview.engine.pivot_align import (
    PivotAlignment,
    PivotAlignmentError,
    align_for_pivot,
)
from apexview.io.dicom_reader import RadiographImage
from apexview.io.image_reader import load_image
from apexview.io.sensor_info import SensorInfo, read_sensor_info


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


def compose_pivot_frame(
    image_a: np.ndarray, warped_b: np.ndarray, t: float
) -> np.ndarray:
    """Blend two engine arrays for the pivot viewer.

    Pure-numpy presentation math, no analysis. Both inputs are uint8
    grayscale of the same shape (the engine guarantees ``warped_b`` has
    the same H,W as ``image_a``). ``t`` is clamped to ``[0.0, 1.0]``: at
    ``0`` returns ``image_a``, at ``1`` returns ``warped_b``, in between
    a linear mix. The viewer slider drives ``t``; the engine arrays are
    not modified.
    """
    if image_a.shape != warped_b.shape:
        raise ValueError(
            f"compose_pivot_frame requires same shape, got "
            f"{image_a.shape} vs {warped_b.shape}"
        )
    if image_a.dtype != np.uint8 or warped_b.dtype != np.uint8:
        raise ValueError("compose_pivot_frame requires uint8 arrays")
    t = max(0.0, min(1.0, float(t)))
    if t == 0.0:
        return image_a.copy()
    if t == 1.0:
        return warped_b.copy()
    blended = (
        image_a.astype(np.float32) * (1.0 - t)
        + warped_b.astype(np.float32) * t
    )
    return blended.astype(np.uint8)


def format_pivot_honesty(inlier_count: int, mean_error_px: float) -> str:
    """One-line honesty string for the pivot viewer.

    Lifts ``inlier_count`` and ``mean_alignment_error_px`` straight from a
    :class:`PivotAlignment` and spells out plainly that residual
    misalignment IS the angle difference, not an alignment defect.
    """
    return (
        f"Rough overlap alignment: {inlier_count} inliers, "
        f"mean error {mean_error_px:.3f} px. "
        f"Residual misalignment is the angle difference (parallax), "
        f"not a defect. This is a comparison view, not a stitch."
    )


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


def _array_to_qpixmap(array: np.ndarray):
    """Wrap a uint8 2D grayscale numpy array as a full-resolution QPixmap.

    Scaling for display is done by
    :meth:`ApexViewWindow._fit_label_pixmap` against the live label
    geometry, so this helper returns the engine pixels at their native
    size and leaves layout decisions to the window.
    """
    _QtCore, QtGui, _QtWidgets = _qt()
    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError("preview requires a 2D uint8 array")
    h, w = array.shape
    # Ensure a contiguous buffer so QImage's view is valid for its lifetime.
    contiguous = np.ascontiguousarray(array)
    qimg = QtGui.QImage(
        contiguous.data, w, h, contiguous.strides[0],
        QtGui.QImage.Format.Format_Grayscale8,
    ).copy()  # copy detaches from the numpy buffer
    return QtGui.QPixmap.fromImage(qimg)


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
        self.current_alignment: PivotAlignment | None = None
        self.current_pivot_frame: np.ndarray | None = None

        # Image labels whose stored ``_original_pixmap`` should re-fit on
        # window resize. Populated by :meth:`_build_image_display`.
        self._image_labels: list = []

        owner = self

        class _ResizingMainWindow(QtWidgets.QMainWindow):
            """QMainWindow subclass that re-fits image pixmaps on resize.

            Defined inside ``__init__`` to keep the lazy-Qt import pattern
            intact: nothing at module-import time touches PyQt6.
            """

            def resizeEvent(inner_self, event):  # noqa: N805
                super().resizeEvent(event)
                owner._on_window_resized()

        class _AutoFitLabel(QtWidgets.QLabel):
            """QLabel that asks the owning window to re-fit its pixmap.

            Fires on every layout-driven size change (panel toggles,
            window resize, splitter drags) so the displayed pixmap always
            matches the label's current geometry.
            """

            def resizeEvent(inner_self, event):  # noqa: N805
                super().resizeEvent(event)
                owner._fit_label_pixmap(inner_self)

        self._ResizingMainWindow = _ResizingMainWindow
        self._AutoFitLabel = _AutoFitLabel

        self.window = _ResizingMainWindow()
        self.window.setWindowTitle("ApexView")
        self.window.resize(960, 760)

        # Scroll area as the central widget so the user can scroll the
        # whole window top-to-bottom. setWidgetResizable lets the inner
        # content widget grow horizontally with the viewport.
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.window.setCentralWidget(self.scroll_area)

        central = QtWidgets.QWidget()
        self.scroll_area.setWidget(central)
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

        # Results panel — text rows stay compact at the top of the scroll.
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

        # Stitched-image preview + save (EXTENSION pairs)
        self.stitched_label = self._build_image_display(min_height=320)
        root.addWidget(self.stitched_label, stretch=2)

        self.save_button = QtWidgets.QPushButton("Save stitched radiograph...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save)
        root.addWidget(self.save_button)

        # Pivot viewer (ANGULATION pairs) -----------------------------------
        # Hidden by default. After Analyze on an ANGULATION pair we either
        # show the aligned pivot view (slider + toggle) when align_for_pivot
        # succeeds, or the plain side-by-side fallback when it raises.
        self.pivot_container = QtWidgets.QWidget()
        pivot_layout = QtWidgets.QVBoxLayout(self.pivot_container)
        pivot_layout.setContentsMargins(0, 0, 0, 0)
        self.pivot_image_label = self._build_image_display(min_height=360)
        pivot_layout.addWidget(self.pivot_image_label, stretch=1)

        self.pivot_honesty_label = QtWidgets.QLabel("")
        self.pivot_honesty_label.setWordWrap(True)
        pivot_layout.addWidget(self.pivot_honesty_label)

        pivot_controls = QtWidgets.QHBoxLayout()
        self.pivot_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pivot_slider.setRange(0, 100)
        self.pivot_slider.setValue(0)
        self.pivot_slider.valueChanged.connect(self._on_pivot_slider_changed)
        pivot_controls.addWidget(self.pivot_slider, stretch=1)
        self.pivot_toggle_button = QtWidgets.QPushButton("Toggle A / B")
        self.pivot_toggle_button.clicked.connect(self._on_pivot_toggle)
        pivot_controls.addWidget(self.pivot_toggle_button)
        pivot_layout.addLayout(pivot_controls)

        self.pivot_save_button = QtWidgets.QPushButton("Save current pivot frame...")
        self.pivot_save_button.setEnabled(False)
        self.pivot_save_button.clicked.connect(self._on_pivot_save)
        pivot_layout.addWidget(self.pivot_save_button)

        self.pivot_container.setVisible(False)
        root.addWidget(self.pivot_container, stretch=3)

        # Fallback side-by-side (ANGULATION pair when alignment refuses) ----
        self.pivot_fallback_container = QtWidgets.QWidget()
        fallback_layout = QtWidgets.QVBoxLayout(self.pivot_fallback_container)
        fallback_layout.setContentsMargins(0, 0, 0, 0)
        fallback_row = QtWidgets.QHBoxLayout()
        self.pivot_fallback_label_a = self._build_image_display(min_height=280)
        self.pivot_fallback_label_b = self._build_image_display(min_height=280)
        fallback_row.addWidget(self.pivot_fallback_label_a, stretch=1)
        fallback_row.addWidget(self.pivot_fallback_label_b, stretch=1)
        fallback_layout.addLayout(fallback_row)
        self.pivot_fallback_message = QtWidgets.QLabel("")
        self.pivot_fallback_message.setWordWrap(True)
        fallback_layout.addWidget(self.pivot_fallback_message)
        self.pivot_fallback_container.setVisible(False)
        root.addWidget(self.pivot_fallback_container, stretch=2)

        # Status line
        self.status_label = QtWidgets.QLabel("Ready.")
        root.addWidget(self.status_label)

    # -- image-display helpers ----------------------------------------------
    def _build_image_display(self, min_height: int = 250):
        """Construct an :class:`_AutoFitLabel` for an image panel.

        The label uses an expanding size policy so it grows when the window
        does, but enforces a minimum height so even on small windows the
        image is at least legible (the user scrolls if total content
        exceeds the viewport).
        """
        QtCore, _QtGui, QtWidgets = self._QtCore, self._QtGui, self._QtWidgets
        label = self._AutoFitLabel("")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(min_height)
        label.setMinimumWidth(240)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        label._original_pixmap = None
        self._image_labels.append(label)
        return label

    def _set_image_label(self, label, array: np.ndarray) -> None:
        """Store the engine array as a full-resolution pixmap on ``label``
        and scale it to the label's current geometry."""
        pixmap = _array_to_qpixmap(array)
        label._original_pixmap = pixmap
        self._fit_label_pixmap(label)

    def _clear_image_label(self, label) -> None:
        label._original_pixmap = None
        label.clear()

    def _fit_label_pixmap(self, label) -> None:
        """Re-scale the label's stored full-resolution pixmap to fit.

        Uses ``KeepAspectRatio`` and ``SmoothTransformation`` so we never
        distort the radiograph. Bounded by both the current width and
        height (falling back to minimums when the layout has not yet
        settled) so a wide image does not bleed past a short panel.
        """
        pixmap = getattr(label, "_original_pixmap", None)
        if pixmap is None or pixmap.isNull():
            return
        QtCore = self._QtCore
        target_w = max(label.width(), label.minimumWidth(), 1)
        target_h = max(label.height(), label.minimumHeight(), 1)
        scaled = pixmap.scaled(
            target_w, target_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _refit_all_image_labels(self) -> None:
        for label in self._image_labels:
            self._fit_label_pixmap(label)

    def _on_window_resized(self) -> None:
        self._refit_all_image_labels()

    # -- slot factory --------------------------------------------------------
    def _build_slot(self, label: str, on_click: Callable[[], None]) -> dict:
        _QtCore, _QtGui, QtWidgets = self._QtCore, self._QtGui, self._QtWidgets
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
        sensor = QtWidgets.QLabel("")
        sensor.setWordWrap(True)
        layout.addWidget(sensor)
        preview = self._build_image_display(min_height=260)
        layout.addWidget(preview, stretch=1)
        return {
            "layout": layout, "button": button,
            "name": name, "spacing": spacing, "sensor": sensor, "preview": preview,
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
            sensor = read_sensor_info(path)
            ui["sensor"].setText(sensor.summary)
        except Exception as exc:
            ui["sensor"].setText(f"Sensor info unavailable: {exc}")
        try:
            self._set_image_label(ui["preview"], image.pixels_u8)
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

        # Reset any prior pivot state; the branch below repopulates it.
        self.current_alignment = None
        self.current_pivot_frame = None
        self._clear_image_label(self.pivot_image_label)
        self._clear_image_label(self.pivot_fallback_label_a)
        self._clear_image_label(self.pivot_fallback_label_b)
        self.pivot_honesty_label.setText("")
        self.pivot_fallback_message.setText("")
        self.pivot_save_button.setEnabled(False)
        self.pivot_container.setVisible(False)
        self.pivot_fallback_container.setVisible(False)

        if display.has_stitched_image and display.stitched_image is not None:
            try:
                self._set_image_label(self.stitched_label, display.stitched_image)
            except Exception as exc:
                self._show_error(f"Could not render stitched preview: {exc}")
                self._clear_image_label(self.stitched_label)
        else:
            self._clear_image_label(self.stitched_label)
        self.save_button.setEnabled(display.save_enabled)

        if result.pair_type is PairType.ANGULATION:
            self._present_pivot()

        # Defer one round-trip so any panels we just made visible have had
        # a chance to lay out before we ask their image labels to re-fit.
        self._QtCore.QTimer.singleShot(0, self._refit_all_image_labels)
        self.status_label.setText("Analysis complete.")

    def _present_pivot(self) -> None:
        """Build and show the pivot viewer for the current ANGULATION pair.

        Calls :func:`align_for_pivot` on the loaded image arrays. On
        success: populates the pivot viewer (slider at 0 -> image_a) and
        shows the honesty line. On :class:`PivotAlignmentError`: shows
        both images side by side without a slider and prints the engine's
        message. Never raises into the caller.
        """
        if self.image_a is None or self.image_b is None:
            return
        try:
            alignment = align_for_pivot(
                self.image_a.pixels_u8, self.image_b.pixels_u8
            )
        except PivotAlignmentError as exc:
            self.pivot_fallback_container.setVisible(True)
            try:
                self._set_image_label(
                    self.pivot_fallback_label_a, self.image_a.pixels_u8
                )
                self._set_image_label(
                    self.pivot_fallback_label_b, self.image_b.pixels_u8
                )
            except Exception as render_exc:
                self.pivot_fallback_message.setText(
                    f"Could not render fallback preview: {render_exc}"
                )
                return
            message = (
                f"Could not align this angulation pair for pivot viewing: "
                f"{exc}. Showing the two images side by side instead."
            )
            self.pivot_fallback_message.setText(message)
            self._show_error(message)
            return
        except ValueError as exc:
            self._show_error(f"Engine rejected the inputs: {exc}")
            return
        except Exception as exc:
            self._show_error(f"Unexpected alignment error: {exc}")
            return

        self.current_alignment = alignment
        self.pivot_container.setVisible(True)
        self.pivot_honesty_label.setText(
            format_pivot_honesty(
                alignment.inlier_count, alignment.mean_alignment_error_px
            )
        )
        self.pivot_slider.blockSignals(True)
        self.pivot_slider.setValue(0)
        self.pivot_slider.blockSignals(False)
        self._refresh_pivot_frame(0)
        self.pivot_save_button.setEnabled(True)

    def _refresh_pivot_frame(self, slider_value: int) -> None:
        alignment = self.current_alignment
        if alignment is None:
            return
        t = max(0.0, min(1.0, slider_value / 100.0))
        try:
            frame = compose_pivot_frame(
                alignment.image_a, alignment.warped_b_to_a, t
            )
        except Exception as exc:
            self._show_error(f"Could not compose pivot frame: {exc}")
            return
        self.current_pivot_frame = frame
        try:
            self._set_image_label(self.pivot_image_label, frame)
        except Exception as exc:
            self._show_error(f"Could not render pivot frame: {exc}")

    def _on_pivot_slider_changed(self, value: int) -> None:
        self._refresh_pivot_frame(int(value))

    def _on_pivot_toggle(self) -> None:
        if self.current_alignment is None:
            return
        new_value = 0 if self.pivot_slider.value() >= 50 else 100
        self.pivot_slider.setValue(new_value)

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

    def _on_pivot_save(self) -> None:
        QtWidgets = self._QtWidgets
        if self.current_pivot_frame is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Save current pivot frame", "pivot.png", SAVE_FILE_FILTER,
        )
        if not path:
            return
        try:
            save_stitched_image(self.current_pivot_frame, path)
        except Exception as exc:
            self._show_error(f"Could not save pivot frame: {exc}")
            return
        self.status_label.setText(f"Saved pivot frame to {path}.")

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
