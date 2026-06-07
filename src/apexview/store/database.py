"""Local SQLite persistence layer for ApexView.

Single-machine, single-user storage of patients, the radiographs they
have IMPORTED into the app, and the analyses run on those radiographs.
There is no networking, no cloud sync, and NO live sensor capture — the
only way a radiograph enters the app is by importing an existing
DICOM/JPEG/PNG file the sensor's own capture software has already
exported. :func:`Database.import_radiograph` COPIES the user's source
file into ApexView's own data directory; the original on the user's disk
is never moved or modified.

Single source of truth: this layer STORES what the engine and readers
already produced. ``verdict``, ``inlier_count``, ``mean_reproj_error``,
and ``message`` on an analysis row are recorded EXACTLY as
:func:`apexview.engine.pair_classifier.classify_pair` returned them
(``NaN`` reprojection error is normalized to SQL ``NULL`` for clean
round-tripping). ``pixel_spacing_source`` and ``sensor_summary`` on a
radiograph row are taken verbatim from
:func:`apexview.io.image_reader.load_image` and
:func:`apexview.io.sensor_info.read_sensor_info`. Nothing here recomputes
geometry, calibration, or device metadata.

Storage location is portable: ``APEXVIEW_DATA_DIR`` overrides everything
(used by tests to redirect to ``tmp_path``); otherwise the per-user app
dir is ``~/Library/Application Support/ApexView`` on macOS and
``~/.apexview`` elsewhere. The SQLite file lives at
``<data-dir>/apexview.sqlite3`` and imported radiograph copies live
under ``<data-dir>/images/`` — nothing is scattered elsewhere on disk.
"""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from apexview.io.image_reader import load_image
from apexview.io.sensor_info import read_sensor_info


# -- Storage path resolution -------------------------------------------------


def _app_data_dir() -> Path:
    """Return the per-user ApexView data directory.

    Honors the ``APEXVIEW_DATA_DIR`` environment variable when set so
    tests can redirect storage to a tmp path. Otherwise falls back to
    the OS-specific per-user app folder.
    """
    env = os.environ.get("APEXVIEW_DATA_DIR")
    if env:
        return Path(os.path.expanduser(env))
    if sys.platform == "darwin":
        return Path(os.path.expanduser("~/Library/Application Support/ApexView"))
    return Path(os.path.expanduser("~/.apexview"))


def db_default_path() -> Path:
    """Return the default SQLite database path, creating the dir if needed."""
    base = _app_data_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "apexview.sqlite3"


# -- Row dataclasses ---------------------------------------------------------


@dataclass(frozen=True)
class Patient:
    id: int
    name: str
    date_of_birth: str | None
    notes: str | None
    created_at: str

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> Patient:
        return cls(
            id=row["id"],
            name=row["name"],
            date_of_birth=row["date_of_birth"],
            notes=row["notes"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Radiograph:
    id: int
    patient_id: int
    original_filename: str | None
    stored_path: str
    imported_at: str
    pixel_spacing_source: str | None
    sensor_summary: str | None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> Radiograph:
        return cls(
            id=row["id"],
            patient_id=row["patient_id"],
            original_filename=row["original_filename"],
            stored_path=row["stored_path"],
            imported_at=row["imported_at"],
            pixel_spacing_source=row["pixel_spacing_source"],
            sensor_summary=row["sensor_summary"],
        )


@dataclass(frozen=True)
class Analysis:
    id: int
    patient_id: int
    radiograph_a_id: int | None
    radiograph_b_id: int | None
    verdict: str | None
    inlier_count: int | None
    mean_reproj_error: float | None
    message: str | None
    result_image_path: str | None
    created_at: str

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> Analysis:
        return cls(
            id=row["id"],
            patient_id=row["patient_id"],
            radiograph_a_id=row["radiograph_a_id"],
            radiograph_b_id=row["radiograph_b_id"],
            verdict=row["verdict"],
            inlier_count=row["inlier_count"],
            mean_reproj_error=row["mean_reproj_error"],
            message=row["message"],
            result_image_path=row["result_image_path"],
            created_at=row["created_at"],
        )


def format_analysis_summary(
    analysis: Analysis,
    name_a: str | None = None,
    name_b: str | None = None,
) -> str:
    """Compose a one-line summary of a past analysis for the history list.

    Uses the verdict and timestamp verbatim from the row and falls back
    to ``#<id>`` for radiographs whose file has since been deleted, so a
    pruned history still reads honestly.
    """
    parts: list[str] = [analysis.created_at, analysis.verdict or "?"]
    if analysis.inlier_count is not None:
        parts.append(f"inliers={analysis.inlier_count}")
    a_label = name_a if name_a else (
        f"#{analysis.radiograph_a_id}" if analysis.radiograph_a_id else "?"
    )
    b_label = name_b if name_b else (
        f"#{analysis.radiograph_b_id}" if analysis.radiograph_b_id else "?"
    )
    parts.append(f"A: {a_label} vs B: {b_label}")
    return " | ".join(parts)


# -- Database ---------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date_of_birth TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS radiographs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    original_filename TEXT,
    stored_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    pixel_spacing_source TEXT,
    sensor_summary TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    radiograph_a_id INTEGER,
    radiograph_b_id INTEGER,
    verdict TEXT,
    inlier_count INTEGER,
    mean_reproj_error REAL,
    message TEXT,
    result_image_path TEXT,
    created_at TEXT NOT NULL
);
"""


class Database:
    """SQLite-backed store for ApexView patients/radiographs/analyses.

    Idempotent: opening an existing database leaves rows intact and only
    creates missing tables. Foreign keys are enabled on every connection
    so ``ON DELETE CASCADE`` actually fires.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        if path is None:
            path = db_default_path()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.path.parent / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- patients --------------------------------------------------------
    def add_patient(
        self,
        name: str,
        date_of_birth: str | None = None,
        notes: str | None = None,
    ) -> Patient:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Patient name must be a non-empty string")
        clean_name = name.strip()
        created_at = _now_iso()
        cur = self._conn.execute(
            "INSERT INTO patients (name, date_of_birth, notes, created_at) "
            "VALUES (?, ?, ?, ?)",
            (clean_name, date_of_birth, notes, created_at),
        )
        self._conn.commit()
        patient = self.get_patient(int(cur.lastrowid))
        assert patient is not None  # we just inserted it
        return patient

    def list_patients(self) -> list[Patient]:
        rows = self._conn.execute(
            "SELECT * FROM patients ORDER BY name COLLATE NOCASE ASC, id ASC"
        ).fetchall()
        return [Patient._from_row(r) for r in rows]

    def get_patient(self, patient_id: int) -> Patient | None:
        row = self._conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        return Patient._from_row(row) if row else None

    def update_patient(
        self,
        patient_id: int,
        name: str | None = None,
        date_of_birth: str | None = None,
        notes: str | None = None,
    ) -> Patient | None:
        existing = self.get_patient(patient_id)
        if existing is None:
            return None
        new_name = existing.name if name is None else name.strip()
        if not new_name:
            raise ValueError("Patient name must be a non-empty string")
        new_dob = existing.date_of_birth if date_of_birth is None else date_of_birth
        new_notes = existing.notes if notes is None else notes
        self._conn.execute(
            "UPDATE patients SET name = ?, date_of_birth = ?, notes = ? WHERE id = ?",
            (new_name, new_dob, new_notes, patient_id),
        )
        self._conn.commit()
        return self.get_patient(patient_id)

    def delete_patient(self, patient_id: int) -> None:
        """Delete the patient and cascade to their radiographs and analyses.

        Also removes the patient's stored radiograph copies (and any saved
        analysis result files that live under our data dir) from disk, so
        a deleted patient leaves no orphan files behind.
        """
        radios = self.list_radiographs(patient_id)
        analyses = self.list_analyses(patient_id)
        self._conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        self._conn.commit()
        for r in radios:
            _unlink_quiet(r.stored_path)
        for a in analyses:
            if a.result_image_path:
                _unlink_quiet(a.result_image_path)

    # -- radiographs -----------------------------------------------------
    def import_radiograph(
        self, patient_id: int, source_path: str | os.PathLike[str]
    ) -> Radiograph:
        """Copy a user's image file into our store and record metadata.

        Reads pixel-spacing source via :func:`load_image` and the sensor
        summary via :func:`read_sensor_info` — both are existing readers
        and remain the only source of those values. The original file on
        the user's disk is NEVER moved or modified.
        """
        if self.get_patient(patient_id) is None:
            raise ValueError(f"unknown patient_id {patient_id}")
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"source file not found: {source}")

        # Existing readers compute the values we record verbatim.
        radiograph = load_image(str(source))
        pixel_spacing_source = radiograph.pixel_spacing_source
        try:
            sensor = read_sensor_info(source)
            sensor_summary = sensor.summary
        except Exception:
            sensor_summary = None

        suffix = source.suffix
        dest_name = f"{uuid.uuid4().hex}{suffix}"
        dest = self.images_dir / dest_name
        shutil.copy2(str(source), str(dest))

        imported_at = _now_iso()
        cur = self._conn.execute(
            "INSERT INTO radiographs "
            "(patient_id, original_filename, stored_path, imported_at, "
            " pixel_spacing_source, sensor_summary) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                patient_id,
                source.name,
                str(dest),
                imported_at,
                pixel_spacing_source,
                sensor_summary,
            ),
        )
        self._conn.commit()
        radio = self.get_radiograph(int(cur.lastrowid))
        assert radio is not None
        return radio

    def list_radiographs(self, patient_id: int) -> list[Radiograph]:
        rows = self._conn.execute(
            "SELECT * FROM radiographs WHERE patient_id = ? "
            "ORDER BY imported_at ASC, id ASC",
            (patient_id,),
        ).fetchall()
        return [Radiograph._from_row(r) for r in rows]

    def get_radiograph(self, radiograph_id: int) -> Radiograph | None:
        row = self._conn.execute(
            "SELECT * FROM radiographs WHERE id = ?", (radiograph_id,)
        ).fetchone()
        return Radiograph._from_row(row) if row else None

    def delete_radiograph(self, radiograph_id: int) -> None:
        radio = self.get_radiograph(radiograph_id)
        if radio is None:
            return
        self._conn.execute(
            "DELETE FROM radiographs WHERE id = ?", (radiograph_id,)
        )
        self._conn.commit()
        _unlink_quiet(radio.stored_path)

    # -- analyses --------------------------------------------------------
    def save_analysis(
        self,
        patient_id: int,
        radiograph_a_id: int | None,
        radiograph_b_id: int | None,
        verdict: str | None,
        inlier_count: int | None,
        mean_reproj_error: float | None,
        message: str | None,
        result_image_path: str | os.PathLike[str] | None = None,
    ) -> Analysis:
        """Persist a single analysis row exactly as the engine reported it.

        ``mean_reproj_error == NaN`` is normalized to SQL ``NULL`` so a
        subsequent round-trip is honest about "no error available"
        without leaning on ``NaN`` survival through sqlite3.
        """
        err: float | None = mean_reproj_error
        if err is not None and isinstance(err, float) and math.isnan(err):
            err = None
        created_at = _now_iso()
        result_path_str = str(result_image_path) if result_image_path else None
        cur = self._conn.execute(
            "INSERT INTO analyses "
            "(patient_id, radiograph_a_id, radiograph_b_id, verdict, "
            " inlier_count, mean_reproj_error, message, result_image_path, "
            " created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient_id,
                radiograph_a_id,
                radiograph_b_id,
                verdict,
                inlier_count,
                err,
                message,
                result_path_str,
                created_at,
            ),
        )
        self._conn.commit()
        analysis = self.get_analysis(int(cur.lastrowid))
        assert analysis is not None
        return analysis

    def get_analysis(self, analysis_id: int) -> Analysis | None:
        row = self._conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        return Analysis._from_row(row) if row else None

    def list_analyses(self, patient_id: int) -> list[Analysis]:
        """Return analyses for ``patient_id`` newest-first."""
        rows = self._conn.execute(
            "SELECT * FROM analyses WHERE patient_id = ? "
            "ORDER BY id DESC",
            (patient_id,),
        ).fetchall()
        return [Analysis._from_row(r) for r in rows]

    def set_analysis_result_path(
        self, analysis_id: int, result_image_path: str | os.PathLike[str] | None
    ) -> Analysis | None:
        path_str = str(result_image_path) if result_image_path else None
        self._conn.execute(
            "UPDATE analyses SET result_image_path = ? WHERE id = ?",
            (path_str, analysis_id),
        )
        self._conn.commit()
        return self.get_analysis(analysis_id)


# -- helpers ----------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _unlink_quiet(path: str | os.PathLike[str]) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        # The file was outside our control (network share, permissions
        # error, already gone). Surfacing this would block a delete the
        # user already confirmed; we record the DB delete and move on.
        pass
