"""Tests for the shared CLAHE preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from apexview.engine.preprocessing import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    preprocess_for_matching,
)


def _gradient_image(h: int = 64, w: int = 96) -> np.ndarray:
    """Non-uniform image so CLAHE has something to rescale."""
    row = np.linspace(20, 230, w, dtype=np.float32)
    img = np.tile(row, (h, 1))
    img[: h // 2, :] *= 0.6  # darker top half -> guaranteed regional variation
    return img.astype(np.uint8)


def test_clahe_on_returns_uint8_same_shape_but_different_pixels():
    img = _gradient_image()
    out = preprocess_for_matching(img, apply_clahe=True)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img), (
        "CLAHE produced identical pixels on a non-uniform image; "
        "preprocessing may be silently a no-op"
    )


def test_clahe_off_is_passthrough():
    img = _gradient_image()
    out = preprocess_for_matching(img, apply_clahe=False)
    assert np.array_equal(out, img)


def test_rejects_non_2d():
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_for_matching(bad)


def test_rejects_non_uint8():
    bad = _gradient_image().astype(np.uint16)
    with pytest.raises(ValueError):
        preprocess_for_matching(bad)


def test_rejects_non_array():
    with pytest.raises(ValueError):
        preprocess_for_matching([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_does_not_mutate_caller_array():
    img = _gradient_image()
    snapshot = img.copy()
    _ = preprocess_for_matching(img, apply_clahe=True)
    assert np.array_equal(img, snapshot), "caller's input was mutated in place"


def test_parameter_constants_are_documented_values():
    # If these change, the docstring rationale must change with them.
    assert CLAHE_CLIP_LIMIT == 2.0
    assert CLAHE_TILE_GRID == (8, 8)
