"""Tests for the detector-plane 2D distance measurement module.

Fully synthetic and deterministic. The anisotropic axis-mapping test is the
critical one: a single isotropic scale or a swapped row/col axis cannot
satisfy all three of its sub-assertions (horizontal, vertical, diagonal).
"""

from __future__ import annotations

import math

import pytest

from apexview.engine.measurement import (
    Measurement,
    ScaleCalibration,
    measure_distance,
)


TIGHT = 1e-9


# --------------------------------------------------------------------------
# Test 1 — isotropic known-answer (3-4-5)
# --------------------------------------------------------------------------
def test_isotropic_reference_length_known_answer():
    calib = ScaleCalibration.from_reference_length(known_mm=50.0, measured_px=100.0)
    assert calib.mm_per_px_col == pytest.approx(0.5, abs=TIGHT)
    assert calib.mm_per_px_row == pytest.approx(0.5, abs=TIGHT)

    m = measure_distance((0.0, 0.0), (30.0, 40.0), calib)
    assert m.distance_px == pytest.approx(50.0, abs=TIGHT)
    assert m.distance_mm == pytest.approx(25.0, abs=TIGHT)


# --------------------------------------------------------------------------
# Test 2 — anisotropic axis-mapping known-answer (THE critical test)
# --------------------------------------------------------------------------
def test_anisotropic_sensor_size_axis_mapping_known_answers():
    calib = ScaleCalibration.from_sensor_size(
        sensor_width_mm=20.0,
        sensor_height_mm=60.0,
        image_shape=(600, 400),  # (rows, cols) == (height_px, width_px)
    )
    # sensor width (20 mm) -> 400 columns -> 0.05 mm per col
    # sensor height (60 mm) -> 600 rows -> 0.10 mm per row
    assert calib.mm_per_px_col == pytest.approx(0.05, abs=TIGHT)
    assert calib.mm_per_px_row == pytest.approx(0.10, abs=TIGHT)
    assert calib.assumes_isotropic is False
    assert calib.source == "sensor_physical_size"

    # Horizontal segment: 100 px in x, 0 in y -> 100 * 0.05 = 5.0 mm
    horiz = measure_distance((0.0, 0.0), (100.0, 0.0), calib)
    assert horiz.distance_px == pytest.approx(100.0, abs=TIGHT)
    assert horiz.distance_mm == pytest.approx(5.0, abs=TIGHT)

    # Vertical segment: 0 in x, 100 px in y -> 100 * 0.10 = 10.0 mm
    vert = measure_distance((0.0, 0.0), (0.0, 100.0), calib)
    assert vert.distance_px == pytest.approx(100.0, abs=TIGHT)
    assert vert.distance_mm == pytest.approx(10.0, abs=TIGHT)

    # Diagonal: hypot(5.0, 10.0). A single-factor (isotropic) calibration
    # cannot reproduce all three of horiz/vert/diag simultaneously.
    diag = measure_distance((0.0, 0.0), (100.0, 100.0), calib)
    assert diag.distance_px == pytest.approx(math.hypot(100.0, 100.0), abs=TIGHT)
    assert diag.distance_mm == pytest.approx(math.hypot(5.0, 10.0), abs=TIGHT)


# --------------------------------------------------------------------------
# Test 3 — from_reference_length forces the two axes equal
# --------------------------------------------------------------------------
def test_reference_length_forces_isotropic():
    calib = ScaleCalibration.from_reference_length(known_mm=12.5, measured_px=250.0)
    assert calib.assumes_isotropic is True
    assert calib.mm_per_px_row == calib.mm_per_px_col
    assert calib.mm_per_px_row == pytest.approx(0.05, abs=TIGHT)


# --------------------------------------------------------------------------
# Test 4 — honesty: magnification flag, raw px, source provenance
# --------------------------------------------------------------------------
def test_honesty_flags_and_provenance():
    sens = ScaleCalibration.from_sensor_size(20.0, 60.0, (600, 400))
    m1 = measure_distance((10.0, 10.0), (60.0, 10.0), sens)
    assert m1.magnification_corrected is False
    assert m1.distance_px == pytest.approx(50.0, abs=TIGHT)
    assert m1.calibration.source == "sensor_physical_size"
    assert m1.calibration is sens  # the calibration is carried, not copied

    ref = ScaleCalibration.from_reference_length(10.0, 50.0)
    m2 = measure_distance((0.0, 0.0), (3.0, 4.0), ref)
    assert m2.magnification_corrected is False
    assert m2.distance_px == pytest.approx(5.0, abs=TIGHT)
    assert m2.calibration.source == "known_reference_length"


# --------------------------------------------------------------------------
# Test 5 — input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "w, h",
    [
        (0.0, 60.0),
        (-1.0, 60.0),
        (20.0, 0.0),
        (20.0, -5.0),
        (float("inf"), 60.0),
        (20.0, float("nan")),
    ],
)
def test_sensor_size_rejects_bad_dimensions(w, h):
    with pytest.raises(ValueError):
        ScaleCalibration.from_sensor_size(w, h, (600, 400))


@pytest.mark.parametrize(
    "shape",
    [
        (0, 400),
        (600, 0),
        (-1, 400),
        (600, 400, 3),
        (600,),
        "600x400",
    ],
)
def test_sensor_size_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        ScaleCalibration.from_sensor_size(20.0, 60.0, shape)


@pytest.mark.parametrize(
    "known_mm, measured_px",
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (10.0, 0.0),
        (10.0, -5.0),
        (float("nan"), 100.0),
        (10.0, float("inf")),
    ],
)
def test_reference_length_rejects_bad_values(known_mm, measured_px):
    with pytest.raises(ValueError):
        ScaleCalibration.from_reference_length(known_mm, measured_px)


def test_measure_distance_rejects_non_length_2_point():
    calib = ScaleCalibration.from_reference_length(1.0, 1.0)
    with pytest.raises(ValueError):
        measure_distance((1.0, 2.0, 3.0), (0.0, 0.0), calib)
    with pytest.raises(ValueError):
        measure_distance((1.0,), (0.0, 0.0), calib)


def test_measure_distance_rejects_non_finite_coordinate():
    calib = ScaleCalibration.from_reference_length(1.0, 1.0)
    with pytest.raises(ValueError):
        measure_distance((float("nan"), 0.0), (0.0, 0.0), calib)
    with pytest.raises(ValueError):
        measure_distance((0.0, 0.0), (float("inf"), 0.0), calib)


def test_measure_distance_rejects_non_numeric_coordinate():
    calib = ScaleCalibration.from_reference_length(1.0, 1.0)
    with pytest.raises(ValueError):
        measure_distance(("x", 0.0), (0.0, 0.0), calib)  # type: ignore[arg-type]


def test_measure_distance_rejects_non_calibration():
    with pytest.raises(ValueError):
        measure_distance((0.0, 0.0), (1.0, 1.0), "not a calibration")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Test 6 — type / field contract
# --------------------------------------------------------------------------
def test_dataclass_field_contract():
    calib = ScaleCalibration.from_sensor_size(20.0, 60.0, (600, 400))
    assert isinstance(calib, ScaleCalibration)
    assert isinstance(calib.mm_per_px_row, float)
    assert isinstance(calib.mm_per_px_col, float)
    assert isinstance(calib.source, str)
    assert isinstance(calib.assumes_isotropic, bool)

    m = measure_distance((0.0, 0.0), (3.0, 4.0), calib)
    assert isinstance(m, Measurement)
    assert isinstance(m.distance_mm, float)
    assert isinstance(m.distance_px, float)
    assert isinstance(m.calibration, ScaleCalibration)
    assert isinstance(m.magnification_corrected, bool)
    assert math.isfinite(m.distance_mm) and math.isfinite(m.distance_px)


def test_dataclasses_are_frozen():
    calib = ScaleCalibration.from_reference_length(1.0, 1.0)
    with pytest.raises(Exception):
        calib.mm_per_px_row = 999.0  # type: ignore[misc]
    m = measure_distance((0.0, 0.0), (1.0, 1.0), calib)
    with pytest.raises(Exception):
        m.distance_mm = 0.0  # type: ignore[misc]
