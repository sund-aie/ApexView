"""PyQt6 desktop GUI for ApexView with local patient management.

THIN CLIENT. The GUI performs NO image analysis. It calls the existing
engine, readers, and store layer and only displays what they already
produced:

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
  * :class:`apexview.store.database.Database` for LOCAL persistence of
    patients, the radiographs they have IMPORTED into the app, and the
    analyses already run on those radiographs. There is no networking,
    no cloud sync, and NO live sensor capture: a radiograph enters the
    app only by importing an existing image file the sensor's own
    capture software has exported.

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
from apexview.store.database import (
    Analysis,
    Database,
    Patient,
    Radiograph,
    format_analysis_summary,
)


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

    def __init__(self, database: Database | None = None) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._QtWidgets = QtWidgets

        # Local persistence. Pass a custom Database in tests or scripted
        # use; otherwise we open the per-user default location.
        self.db: Database = database if database is not None else Database()

        # Selected-patient state
        self.current_patient: Patient | None = None
        self.current_radiographs: list[Radiograph] = []
        self.current_analyses: list[Analysis] = []

        # Loaded image state for the Analyze flow
        self.image_a: RadiographImage | None = None
        self.image_b: RadiographImage | None = None
        self.radiograph_a: Radiograph | None = None
        self.radiograph_b: Radiograph | None = None

        # Analyze result state
        self.last_display: AnalysisDisplay | None = None
        self.current_alignment: PivotAlignment | None = None
        self.current_pivot_frame: np.ndarray | None = None
        # Row id of the most recently persisted analysis so a follow-up
        # Save stitched / Save pivot frame call can back-fill its
        # result_image_path column.
        self.current_analysis_id: int | None = None

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
        self.window.resize(1180, 820)

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
        top_row = QtWidgets.QHBoxLayout(central)

        # -- Left column: patients + history ----------------------------
        left_column = QtWidgets.QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_holder = QtWidgets.QWidget()
        left_holder.setLayout(left_column)
        left_holder.setMinimumWidth(280)
        left_holder.setMaximumWidth(340)
        top_row.addWidget(left_holder)

        left_column.addWidget(self._section_label("Patients"))
        self.patient_list = QtWidgets.QListWidget()
        self.patient_list.itemSelectionChanged.connect(
            self._on_patient_selection_changed
        )
        left_column.addWidget(self.patient_list, stretch=1)

        patient_button_row = QtWidgets.QHBoxLayout()
        self.add_patient_button = QtWidgets.QPushButton("Add Patient...")
        self.add_patient_button.clicked.connect(self._on_add_patient)
        self.delete_patient_button = QtWidgets.QPushButton("Delete Patient")
        self.delete_patient_button.setEnabled(False)
        self.delete_patient_button.clicked.connect(self._on_delete_patient)
        patient_button_row.addWidget(self.add_patient_button)
        patient_button_row.addWidget(self.delete_patient_button)
        left_column.addLayout(patient_button_row)

        left_column.addWidget(self._section_label("Analysis history"))
        self.history_list = QtWidgets.QListWidget()
        left_column.addWidget(self.history_list, stretch=1)

        # -- Right column: workflow -------------------------------------
        right_column = QtWidgets.QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_holder = QtWidgets.QWidget()
        right_holder.setLayout(right_column)
        top_row.addWidget(right_holder, stretch=1)

        self.patient_header_label = QtWidgets.QLabel("No patient selected.")
        header_font = self.patient_header_label.font()
        header_font.setPointSize(header_font.pointSize() + 4)
        header_font.setBold(True)
        self.patient_header_label.setFont(header_font)
        right_column.addWidget(self.patient_header_label)

        self.patient_details_label = QtWidgets.QLabel("")
        self.patient_details_label.setWordWrap(True)
        right_column.addWidget(self.patient_details_label)

        # Radiographs for the selected patient
        right_column.addWidget(self._section_label("Imported radiographs"))
        self.radiograph_list = QtWidgets.QListWidget()
        self.radiograph_list.setMinimumHeight(120)
        right_column.addWidget(self.radiograph_list)

        radio_button_row = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import Radiograph...")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._on_import_radiograph)
        self.delete_radiograph_button = QtWidgets.QPushButton("Delete Selected Radiograph")
        self.delete_radiograph_button.setEnabled(False)
        self.delete_radiograph_button.clicked.connect(self._on_delete_radiograph)
        radio_button_row.addWidget(self.import_button)
        radio_button_row.addWidget(self.delete_radiograph_button)
        radio_button_row.addStretch(1)
        right_column.addLayout(radio_button_row)
        self.radiograph_list.itemSelectionChanged.connect(
            self._on_radiograph_list_selection_changed
        )

        # Image A/B selection
        right_column.addWidget(self._section_label("Select images to compare"))
        pair_row = QtWidgets.QHBoxLayout()
        right_column.addLayout(pair_row)
        self._slot_a = self._build_slot("A")
        self._slot_b = self._build_slot("B")
        pair_row.addLayout(self._slot_a["layout"])
        pair_row.addLayout(self._slot_b["layout"])

        # Analyze button
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._on_analyze)
        right_column.addWidget(self.analyze_button)

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
        for w in (
            self.verdict_label, self.inlier_label,
            self.reproj_label, self.message_label,
        ):
            right_column.addWidget(w)

        # Stitched-image preview + save (EXTENSION pairs)
        self.stitched_label = self._build_image_display(min_height=320)
        right_column.addWidget(self.stitched_label, stretch=2)

        self.save_button = QtWidgets.QPushButton("Save stitched radiograph...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save)
        right_column.addWidget(self.save_button)

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
        right_column.addWidget(self.pivot_container, stretch=3)

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
        right_column.addWidget(self.pivot_fallback_container, stretch=2)

        # Status line
        self.status_label = QtWidgets.QLabel(
            "Ready. Add or select a patient to begin."
        )
        right_column.addWidget(self.status_label)

        # Initial population from disk.
        self._refresh_patient_list()

    # -- section header helper ----------------------------------------------
    def _section_label(self, text: str):
        QtWidgets = self._QtWidgets
        label = QtWidgets.QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

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
    def _build_slot(self, label: str) -> dict:
        _QtCore, _QtGui, QtWidgets = self._QtCore, self._QtGui, self._QtWidgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(f"Image {label}"))
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(
            lambda _index, slot=label: self._on_slot_combo_changed(slot)
        )
        layout.addWidget(combo)
        name = QtWidgets.QLabel("(no image selected)")
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
            "layout": layout, "combo": combo,
            "name": name, "spacing": spacing, "sensor": sensor, "preview": preview,
        }

    # -- patient list management --------------------------------------------
    def _refresh_patient_list(self, select_patient_id: int | None = None) -> None:
        self.patient_list.blockSignals(True)
        self.patient_list.clear()
        patients = self.db.list_patients()
        row_to_select = -1
        for i, patient in enumerate(patients):
            item = self._QtWidgets.QListWidgetItem(patient.name)
            item.setData(self._QtCore.Qt.ItemDataRole.UserRole, patient.id)
            self.patient_list.addItem(item)
            if patient.id == select_patient_id:
                row_to_select = i
        self.patient_list.blockSignals(False)
        if row_to_select >= 0:
            self.patient_list.setCurrentRow(row_to_select)
        else:
            self._on_patient_selection_changed()

    def _selected_patient_id(self) -> int | None:
        item = self.patient_list.currentItem()
        if item is None:
            return None
        value = item.data(self._QtCore.Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _on_patient_selection_changed(self) -> None:
        pid = self._selected_patient_id()
        if pid is None:
            self.current_patient = None
            self.current_radiographs = []
            self.current_analyses = []
            self.patient_header_label.setText("No patient selected.")
            self.patient_details_label.setText("")
            self.radiograph_list.clear()
            self.history_list.clear()
            self.delete_patient_button.setEnabled(False)
            self.import_button.setEnabled(False)
            self.delete_radiograph_button.setEnabled(False)
            self._populate_slot_combos()
            self._refresh_analyze_button()
            return

        self.current_patient = self.db.get_patient(pid)
        if self.current_patient is None:
            self._show_error(f"Patient {pid} is no longer in the database.")
            self._refresh_patient_list()
            return

        self.patient_header_label.setText(f"Patient: {self.current_patient.name}")
        bits: list[str] = []
        if self.current_patient.date_of_birth:
            bits.append(f"DOB: {self.current_patient.date_of_birth}")
        if self.current_patient.notes:
            bits.append(self.current_patient.notes)
        self.patient_details_label.setText("  |  ".join(bits))

        self.delete_patient_button.setEnabled(True)
        self.import_button.setEnabled(True)

        # Reset Analyze-pane state for the new patient.
        self.image_a = None
        self.image_b = None
        self.radiograph_a = None
        self.radiograph_b = None
        self._reset_slot(self._slot_a)
        self._reset_slot(self._slot_b)
        self._reset_results_panel()

        self._refresh_radiograph_list()
        self._refresh_history_list()
        self._refresh_analyze_button()
        self.status_label.setText(
            f"Selected patient: {self.current_patient.name}."
        )

    def _refresh_radiograph_list(self) -> None:
        self.radiograph_list.clear()
        self.current_radiographs = []
        if self.current_patient is None:
            self._populate_slot_combos()
            return
        radios = self.db.list_radiographs(self.current_patient.id)
        self.current_radiographs = radios
        for radio in radios:
            text = radio.original_filename or Path(radio.stored_path).name
            if radio.sensor_summary:
                text += f"\n  {radio.sensor_summary}"
            item = self._QtWidgets.QListWidgetItem(text)
            item.setData(self._QtCore.Qt.ItemDataRole.UserRole, radio.id)
            self.radiograph_list.addItem(item)
        self._populate_slot_combos()
        self.delete_radiograph_button.setEnabled(False)

    def _on_radiograph_list_selection_changed(self) -> None:
        self.delete_radiograph_button.setEnabled(
            self.radiograph_list.currentItem() is not None
        )

    def _populate_slot_combos(self) -> None:
        for slot in (self._slot_a, self._slot_b):
            combo = slot["combo"]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(select)", None)
            for radio in self.current_radiographs:
                label = radio.original_filename or Path(radio.stored_path).name
                combo.addItem(label, radio.id)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def _refresh_history_list(self) -> None:
        self.history_list.clear()
        self.current_analyses = []
        if self.current_patient is None:
            return
        analyses = self.db.list_analyses(self.current_patient.id)
        self.current_analyses = analyses
        radio_by_id = {r.id: r for r in self.current_radiographs}
        for analysis in analyses:
            radio_a = radio_by_id.get(analysis.radiograph_a_id or -1)
            radio_b = radio_by_id.get(analysis.radiograph_b_id or -1)
            name_a = radio_a.original_filename if radio_a else None
            name_b = radio_b.original_filename if radio_b else None
            text = format_analysis_summary(analysis, name_a=name_a, name_b=name_b)
            self.history_list.addItem(text)

    def _reset_slot(self, slot: dict) -> None:
        slot["combo"].blockSignals(True)
        slot["combo"].setCurrentIndex(0)
        slot["combo"].blockSignals(False)
        slot["name"].setText("(no image selected)")
        slot["spacing"].setText("")
        slot["sensor"].setText("")
        self._clear_image_label(slot["preview"])

    def _reset_results_panel(self) -> None:
        self.last_display = None
        self.current_alignment = None
        self.current_pivot_frame = None
        self.current_analysis_id = None
        self.verdict_label.setText("")
        self.inlier_label.setText("")
        self.reproj_label.setText("")
        self.message_label.setText("")
        self._clear_image_label(self.stitched_label)
        self._clear_image_label(self.pivot_image_label)
        self._clear_image_label(self.pivot_fallback_label_a)
        self._clear_image_label(self.pivot_fallback_label_b)
        self.pivot_honesty_label.setText("")
        self.pivot_fallback_message.setText("")
        self.save_button.setEnabled(False)
        self.pivot_save_button.setEnabled(False)
        self.pivot_container.setVisible(False)
        self.pivot_fallback_container.setVisible(False)

    def _refresh_analyze_button(self) -> None:
        ready = (
            self.current_patient is not None
            and self.image_a is not None
            and self.image_b is not None
        )
        self.analyze_button.setEnabled(ready)

    # -- patient actions ----------------------------------------------------
    def _on_add_patient(self) -> None:
        result = self._prompt_new_patient()
        if result is None:
            return
        name, dob, notes = result
        try:
            patient = self.db.add_patient(name, date_of_birth=dob, notes=notes)
        except ValueError as exc:
            self._show_error(f"Could not add patient: {exc}")
            return
        except Exception as exc:
            self._show_error(f"Unexpected error adding patient: {exc}")
            return
        self._refresh_patient_list(select_patient_id=patient.id)
        self.status_label.setText(f"Added patient: {patient.name}.")

    def _prompt_new_patient(self) -> tuple[str, str | None, str | None] | None:
        QtWidgets = self._QtWidgets
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle("Add Patient")
        layout = QtWidgets.QFormLayout(dialog)
        name_edit = QtWidgets.QLineEdit()
        dob_edit = QtWidgets.QLineEdit()
        dob_edit.setPlaceholderText("YYYY-MM-DD (optional)")
        notes_edit = QtWidgets.QPlainTextEdit()
        notes_edit.setPlaceholderText("Notes (optional)")
        notes_edit.setFixedHeight(80)
        layout.addRow("Name *:", name_edit)
        layout.addRow("DOB:", dob_edit)
        layout.addRow("Notes:", notes_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        name = name_edit.text().strip()
        if not name:
            self._show_error("Patient name is required.")
            return None
        dob = dob_edit.text().strip() or None
        notes = notes_edit.toPlainText().strip() or None
        return (name, dob, notes)

    def _on_delete_patient(self) -> None:
        if self.current_patient is None:
            return
        QtWidgets = self._QtWidgets
        answer = QtWidgets.QMessageBox.question(
            self.window,
            "Delete patient",
            (
                f"Delete patient '{self.current_patient.name}' and all "
                f"their imported radiographs and saved analyses? This "
                f"removes the stored files from disk and cannot be undone."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        deleted_name = self.current_patient.name
        try:
            self.db.delete_patient(self.current_patient.id)
        except Exception as exc:
            self._show_error(f"Could not delete patient: {exc}")
            return
        self.current_patient = None
        self._refresh_patient_list()
        self.status_label.setText(f"Deleted patient: {deleted_name}.")

    # -- radiograph actions -------------------------------------------------
    def _on_import_radiograph(self) -> None:
        if self.current_patient is None:
            self._show_error("Select a patient before importing a radiograph.")
            return
        QtWidgets = self._QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Import radiograph", "", SUPPORTED_FILE_FILTER,
        )
        if not path:
            return
        try:
            radio = self.db.import_radiograph(self.current_patient.id, path)
        except FileNotFoundError as exc:
            self._show_error(f"Could not import radiograph: {exc}")
            return
        except ValueError as exc:
            self._show_error(f"Could not import radiograph: {exc}")
            return
        except Exception as exc:
            self._show_error(f"Unexpected error importing radiograph: {exc}")
            return
        self._refresh_radiograph_list()
        self.status_label.setText(
            f"Imported {radio.original_filename} for "
            f"{self.current_patient.name}."
        )

    def _on_delete_radiograph(self) -> None:
        item = self.radiograph_list.currentItem()
        if item is None or self.current_patient is None:
            return
        radio_id = item.data(self._QtCore.Qt.ItemDataRole.UserRole)
        QtWidgets = self._QtWidgets
        answer = QtWidgets.QMessageBox.question(
            self.window,
            "Delete radiograph",
            (
                "Delete the selected radiograph for this patient? The "
                "stored copy will be removed from disk; the original on "
                "your computer is untouched."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.db.delete_radiograph(int(radio_id))
        except Exception as exc:
            self._show_error(f"Could not delete radiograph: {exc}")
            return
        # Anything previously loaded that pointed at this radiograph is
        # now stale; reset the Analyze pane defensively.
        self.image_a = None
        self.image_b = None
        self.radiograph_a = None
        self.radiograph_b = None
        self._reset_slot(self._slot_a)
        self._reset_slot(self._slot_b)
        self._reset_results_panel()
        self._refresh_radiograph_list()
        self._refresh_history_list()
        self._refresh_analyze_button()
        self.status_label.setText("Radiograph deleted.")

    # -- image selection ----------------------------------------------------
    def _on_slot_combo_changed(self, slot_name: str) -> None:
        slot = self._slot_a if slot_name == "A" else self._slot_b
        combo = slot["combo"]
        radio_id = combo.currentData()
        if radio_id is None:
            self._reset_slot(slot)
            if slot_name == "A":
                self.image_a = None
                self.radiograph_a = None
            else:
                self.image_b = None
                self.radiograph_b = None
            self._refresh_analyze_button()
            return
        radio = self.db.get_radiograph(int(radio_id))
        if radio is None:
            self._show_error(
                f"Radiograph {radio_id} is no longer in the database."
            )
            self._reset_slot(slot)
            self._refresh_radiograph_list()
            return
        try:
            image = load_image(radio.stored_path)
        except (FileNotFoundError, ValueError) as exc:
            self._show_error(
                f"Could not load image {slot_name}: {exc}"
            )
            self._reset_slot(slot)
            return
        slot["name"].setText(
            radio.original_filename or Path(radio.stored_path).name
        )
        slot["spacing"].setText(format_pixel_spacing(image))
        slot["sensor"].setText(radio.sensor_summary or "Sensor info unavailable.")
        try:
            self._set_image_label(slot["preview"], image.pixels_u8)
        except Exception as exc:
            self._show_error(
                f"Could not render preview for {slot_name}: {exc}"
            )
        if slot_name == "A":
            self.image_a = image
            self.radiograph_a = radio
        else:
            self.image_b = image
            self.radiograph_b = radio
        self._refresh_analyze_button()
        self.status_label.setText(f"Loaded image {slot_name}.")

    # -- analyze ------------------------------------------------------------
    def _on_analyze(self) -> None:
        if (
            self.current_patient is None
            or self.image_a is None
            or self.image_b is None
        ):
            return
        try:
            result = classify_pair(self.image_a.pixels_u8, self.image_b.pixels_u8)
        except ValueError as exc:
            self._show_error(f"Engine rejected the inputs: {exc}")
            return
        except Exception as exc:
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

        # Persist the analysis exactly as the engine reported it. NaN
        # reprojection error is normalized to NULL inside save_analysis.
        try:
            saved = self.db.save_analysis(
                patient_id=self.current_patient.id,
                radiograph_a_id=self.radiograph_a.id if self.radiograph_a else None,
                radiograph_b_id=self.radiograph_b.id if self.radiograph_b else None,
                verdict=display.verdict_label,
                inlier_count=int(result.inlier_count),
                mean_reproj_error=float(result.mean_reprojection_error),
                message=result.message,
            )
            self.current_analysis_id = saved.id
            self._refresh_history_list()
        except Exception as exc:
            self._show_error(f"Analysis ran but could not be saved: {exc}")

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

    # -- save handlers ------------------------------------------------------
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
        if self.current_analysis_id is not None:
            try:
                self.db.set_analysis_result_path(
                    self.current_analysis_id, path
                )
                self._refresh_history_list()
            except Exception as exc:
                self._show_error(
                    f"Saved image but could not record path in database: {exc}"
                )
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
        if self.current_analysis_id is not None:
            try:
                self.db.set_analysis_result_path(
                    self.current_analysis_id, path
                )
                self._refresh_history_list()
            except Exception as exc:
                self._show_error(
                    f"Saved frame but could not record path in database: {exc}"
                )
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
