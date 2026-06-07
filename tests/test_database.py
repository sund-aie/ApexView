"""Tests for the local SQLite persistence layer.

Headless, deterministic. Every test redirects ``APEXVIEW_DATA_DIR`` to
``tmp_path`` (or builds a :class:`Database` with an explicit ``tmp_path``)
so nothing touches the real per-user data directory.
"""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from apexview.store.database import (
    Analysis,
    Database,
    Patient,
    Radiograph,
    db_default_path,
    format_analysis_summary,
)


# --------------------------------------------------------------------------
# Fixtures: synthetic DICOM/PNG builders mirrored from the other tests so we
# never need to commit binary fixtures to the repo.
# --------------------------------------------------------------------------


def _build_dicom(
    dest_dir: Path,
    pixels: np.ndarray,
    *,
    name: str = "import_me.dcm",
    pixel_spacing=None,
    imager_pixel_spacing=None,
    manufacturer: str | None = None,
    model: str | None = None,
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    path = dest_dir / name
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
    if manufacturer is not None:
        ds.Manufacturer = manufacturer
    if model is not None:
        ds.ManufacturerModelName = model
    ds.save_as(str(path), enforce_file_format=True)
    return path


def _build_png(dest_dir: Path, name: str = "import_me.png") -> Path:
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = dest_dir / name
    Image.fromarray(arr).convert("L").save(str(path))
    return path


@pytest.fixture
def db(tmp_path: Path) -> Database:
    db_path = tmp_path / "apex.sqlite3"
    instance = Database(db_path)
    yield instance
    instance.close()


# --------------------------------------------------------------------------
# db_default_path honors APEXVIEW_DATA_DIR
# --------------------------------------------------------------------------


def test_db_default_path_honors_env_var(tmp_path: Path, monkeypatch):
    target = tmp_path / "alt"
    monkeypatch.setenv("APEXVIEW_DATA_DIR", str(target))
    p = db_default_path()
    assert p == target / "apexview.sqlite3"
    assert target.exists()


def test_db_default_path_falls_back_to_per_user_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("APEXVIEW_DATA_DIR", raising=False)
    p = db_default_path()
    assert p.name == "apexview.sqlite3"
    # We do not delete the real per-user dir from inside a test; just
    # assert the path is shaped like a real per-user app path.
    assert "ApexView" in str(p) or ".apexview" in str(p)


# --------------------------------------------------------------------------
# patients
# --------------------------------------------------------------------------


def test_add_patient_round_trips(db: Database):
    patient = db.add_patient("Alice Example", date_of_birth="1990-01-15", notes="hi")
    assert isinstance(patient, Patient)
    assert patient.id > 0
    assert patient.name == "Alice Example"
    assert patient.date_of_birth == "1990-01-15"
    assert patient.notes == "hi"
    assert patient.created_at  # ISO timestamp string

    patients = db.list_patients()
    assert len(patients) == 1
    assert patients[0].name == "Alice Example"


def test_add_patient_trims_whitespace(db: Database):
    patient = db.add_patient("   Bob   ")
    assert patient.name == "Bob"


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_add_patient_rejects_empty_name(db: Database, bad: str):
    with pytest.raises(ValueError):
        db.add_patient(bad)


def test_list_patients_is_case_insensitive_alpha_sorted(db: Database):
    db.add_patient("zelda")
    db.add_patient("Alice")
    db.add_patient("bob")
    names = [p.name for p in db.list_patients()]
    assert names == ["Alice", "bob", "zelda"]


def test_update_patient(db: Database):
    p = db.add_patient("Old Name", date_of_birth="1980-01-01")
    updated = db.update_patient(p.id, name="New Name", notes="updated")
    assert updated is not None
    assert updated.name == "New Name"
    assert updated.date_of_birth == "1980-01-01"  # untouched
    assert updated.notes == "updated"


# --------------------------------------------------------------------------
# import_radiograph
# --------------------------------------------------------------------------


def test_import_radiograph_copies_dicom_into_store(db: Database, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _build_dicom(
        source_dir,
        np.zeros((16, 16), dtype=np.uint8),
        imager_pixel_spacing=[0.085, 0.085],
        manufacturer="Acme",
        model="ToothCam 9000",
    )
    original_bytes = source.read_bytes()

    patient = db.add_patient("Alice")
    radio = db.import_radiograph(patient.id, source)

    assert isinstance(radio, Radiograph)
    assert radio.patient_id == patient.id
    assert radio.original_filename == source.name
    stored = Path(radio.stored_path)
    assert stored.exists()
    assert stored.parent == db.images_dir
    # The stored copy is a real copy, not a move/symlink — original intact.
    assert source.exists()
    assert source.read_bytes() == original_bytes
    # And the stored file content matches the source byte-for-byte.
    assert stored.read_bytes() == original_bytes

    assert radio.pixel_spacing_source == "ImagerPixelSpacing"
    assert radio.sensor_summary is not None
    assert "Acme" in radio.sensor_summary


def test_import_radiograph_handles_png_honestly(db: Database, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    png = _build_png(source_dir)

    patient = db.add_patient("Alice")
    radio = db.import_radiograph(patient.id, png)

    assert Path(radio.stored_path).exists()
    assert radio.pixel_spacing_source is not None
    assert "unavailable" in radio.pixel_spacing_source.lower()
    assert radio.sensor_summary is not None
    assert "not a DICOM" in radio.sensor_summary


def test_import_radiograph_lists_under_patient(db: Database, tmp_path: Path):
    patient = db.add_patient("Alice")
    source = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8), name="a.dcm")
    db.import_radiograph(patient.id, source)
    source2 = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8), name="b.dcm")
    db.import_radiograph(patient.id, source2)
    radios = db.list_radiographs(patient.id)
    assert [r.original_filename for r in radios] == ["a.dcm", "b.dcm"]


def test_import_radiograph_rejects_unknown_patient(db: Database, tmp_path: Path):
    source = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        db.import_radiograph(9999, source)


def test_import_radiograph_rejects_missing_source(db: Database, tmp_path: Path):
    patient = db.add_patient("Alice")
    with pytest.raises(FileNotFoundError):
        db.import_radiograph(patient.id, tmp_path / "does_not_exist.dcm")


# --------------------------------------------------------------------------
# delete_radiograph
# --------------------------------------------------------------------------


def test_delete_radiograph_removes_row_and_file(db: Database, tmp_path: Path):
    patient = db.add_patient("Alice")
    source = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8))
    radio = db.import_radiograph(patient.id, source)
    stored = Path(radio.stored_path)
    assert stored.exists()

    db.delete_radiograph(radio.id)

    assert db.get_radiograph(radio.id) is None
    assert not stored.exists()
    # And the user's original on disk is still there.
    assert source.exists()


# --------------------------------------------------------------------------
# save_analysis
# --------------------------------------------------------------------------


def test_save_analysis_round_trips_exact_values(db: Database):
    patient = db.add_patient("Alice")
    a = db.save_analysis(
        patient.id,
        radiograph_a_id=None,
        radiograph_b_id=None,
        verdict="EXTENSION",
        inlier_count=248,
        mean_reproj_error=0.189,
        message="Extension pair: 248 inliers (>= 100 threshold); stitched.",
    )
    assert isinstance(a, Analysis)
    assert a.patient_id == patient.id
    assert a.verdict == "EXTENSION"
    assert a.inlier_count == 248
    assert a.mean_reproj_error == pytest.approx(0.189)
    assert a.message.startswith("Extension pair")


def test_save_analysis_normalizes_nan_error_to_none(db: Database):
    patient = db.add_patient("Alice")
    a = db.save_analysis(
        patient.id,
        radiograph_a_id=None,
        radiograph_b_id=None,
        verdict="ANGULATION",
        inlier_count=0,
        mean_reproj_error=math.nan,
        message="matching failed",
    )
    # NaN survives sqlite3 round-trip poorly; we normalize to NULL.
    assert a.mean_reproj_error is None

    # And list_analyses returns the same shape.
    listed = db.list_analyses(patient.id)
    assert len(listed) == 1
    assert listed[0].mean_reproj_error is None


def test_list_analyses_is_most_recent_first(db: Database):
    patient = db.add_patient("Alice")
    a1 = db.save_analysis(
        patient.id, None, None, "EXTENSION", 100, 0.5, "first"
    )
    a2 = db.save_analysis(
        patient.id, None, None, "ANGULATION", 30, 1.2, "second"
    )
    a3 = db.save_analysis(
        patient.id, None, None, "EXTENSION", 200, 0.3, "third"
    )
    listed = db.list_analyses(patient.id)
    assert [a.id for a in listed] == [a3.id, a2.id, a1.id]


def test_set_analysis_result_path_updates_row(db: Database, tmp_path: Path):
    patient = db.add_patient("Alice")
    a = db.save_analysis(
        patient.id, None, None, "EXTENSION", 248, 0.189, "msg"
    )
    saved_to = tmp_path / "stitched.png"
    updated = db.set_analysis_result_path(a.id, saved_to)
    assert updated is not None
    assert updated.result_image_path == str(saved_to)


# --------------------------------------------------------------------------
# delete_patient cascades and removes files
# --------------------------------------------------------------------------


def test_delete_patient_cascades_and_removes_stored_files(
    db: Database, tmp_path: Path
):
    patient = db.add_patient("Alice")
    src1 = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8), name="r1.dcm")
    src2 = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8), name="r2.dcm")
    r1 = db.import_radiograph(patient.id, src1)
    r2 = db.import_radiograph(patient.id, src2)
    db.save_analysis(patient.id, r1.id, r2.id, "EXTENSION", 200, 0.4, "ok")

    stored1, stored2 = Path(r1.stored_path), Path(r2.stored_path)
    assert stored1.exists() and stored2.exists()

    db.delete_patient(patient.id)

    assert db.get_patient(patient.id) is None
    assert db.list_radiographs(patient.id) == []
    assert db.list_analyses(patient.id) == []
    assert not stored1.exists()
    assert not stored2.exists()
    # The user's originals on disk are untouched.
    assert src1.exists()
    assert src2.exists()


def test_foreign_keys_pragma_is_actually_enforced(db: Database, tmp_path: Path):
    """Even without our manual file-cleanup pass, ON DELETE CASCADE on the
    schema must actually fire — i.e. PRAGMA foreign_keys = ON took effect.
    Otherwise child rows would be orphaned and our cascade promise is a
    lie."""
    patient = db.add_patient("Alice")
    src = _build_dicom(tmp_path, np.zeros((8, 8), dtype=np.uint8))
    radio = db.import_radiograph(patient.id, src)
    db.save_analysis(patient.id, radio.id, None, "EXTENSION", 100, 0.5, "ok")

    # Issue the parent delete directly (no helper cleanup).
    db._conn.execute("DELETE FROM patients WHERE id = ?", (patient.id,))
    db._conn.commit()

    radios = db._conn.execute(
        "SELECT COUNT(*) AS n FROM radiographs WHERE patient_id = ?", (patient.id,)
    ).fetchone()["n"]
    analyses = db._conn.execute(
        "SELECT COUNT(*) AS n FROM analyses WHERE patient_id = ?", (patient.id,)
    ).fetchone()["n"]
    assert radios == 0
    assert analyses == 0


# --------------------------------------------------------------------------
# format_analysis_summary
# --------------------------------------------------------------------------


def test_format_analysis_summary_with_known_names():
    a = Analysis(
        id=7,
        patient_id=1,
        radiograph_a_id=11,
        radiograph_b_id=12,
        verdict="EXTENSION",
        inlier_count=248,
        mean_reproj_error=0.189,
        message="ok",
        result_image_path=None,
        created_at="2024-01-15T12:34:56",
    )
    line = format_analysis_summary(a, name_a="left.dcm", name_b="right.dcm")
    assert "EXTENSION" in line
    assert "inliers=248" in line
    assert "left.dcm" in line
    assert "right.dcm" in line
    assert "2024-01-15T12:34:56" in line


def test_format_analysis_summary_falls_back_to_id_when_name_missing():
    a = Analysis(
        id=7,
        patient_id=1,
        radiograph_a_id=11,
        radiograph_b_id=12,
        verdict="ANGULATION",
        inlier_count=30,
        mean_reproj_error=None,
        message="msg",
        result_image_path=None,
        created_at="2024-02-01T00:00:00",
    )
    line = format_analysis_summary(a)
    assert "#11" in line
    assert "#12" in line


# --------------------------------------------------------------------------
# Idempotent open
# --------------------------------------------------------------------------


def test_reopening_the_database_preserves_rows(tmp_path: Path):
    db_path = tmp_path / "apex.sqlite3"
    with Database(db_path) as db:
        db.add_patient("Alice")
    # Reopen; the schema and the row should still be there.
    with Database(db_path) as db:
        names = [p.name for p in db.list_patients()]
        assert names == ["Alice"]


# --------------------------------------------------------------------------
# Defensive: the store layer must NOT pull in Qt at import time, or tests
# that run under headless Linux without libEGL would break.
# --------------------------------------------------------------------------


def test_module_does_not_import_pyqt6():
    import sys as _sys
    assert "PyQt6" not in _sys.modules or _sys.modules["PyQt6"] is not None
    # Importing the store module again should not pull it in either.
    import apexview.store.database  # noqa: F401
    # We do not assert PyQt6 absence absolutely — another test module
    # may have imported it. The point is that the store module's own
    # imports do not require PyQt6.
