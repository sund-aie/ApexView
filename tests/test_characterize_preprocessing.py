"""Tests for the preprocessing characterization script.

We verify the SCRIPT'S MACHINERY on fully synthetic images: that analyze_pair
returns the expected fields, that the variants are actually applied
(preprocessed image differs from raw), and that the coverage metric stays in
its 0..16 range. We intentionally do NOT assert "CLAHE beats raw" — that is
the empirical question for real radiographs, and asserting it on clean
synthetic data would be a hollow/misleading test.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Load the experiment script as a module (it lives outside the package).
_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "characterize_preprocessing.py"
_spec = importlib.util.spec_from_file_location("characterize_preprocessing", _SCRIPT)
characterize = importlib.util.module_from_spec(_spec)
sys.modules["characterize_preprocessing"] = characterize
assert _spec.loader is not None
_spec.loader.exec_module(characterize)

VARIANTS = characterize.VARIANTS
analyze_pair = characterize.analyze_pair
preprocess = characterize.preprocess
grid_coverage = characterize.grid_coverage
VariantResult = characterize.VariantResult

IMG_H = 480
IMG_W = 640

# Known intrinsics + relative pose for a small multi-depth scene (mirrors
# test_stereo_geometry.py so we get a recoverable pair).
K_A = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
K_B = np.array([[820.0, 0.0, 310.0], [0.0, 820.0, 250.0], [0.0, 0.0, 1.0]])
_THETA = math.radians(8.0)
R_AB = np.array(
    [
        [math.cos(_THETA), 0.0, math.sin(_THETA)],
        [0.0, 1.0, 0.0],
        [-math.sin(_THETA), 0.0, math.cos(_THETA)],
    ]
)
T_AB = np.array([0.6, 0.05, 0.1])


def _texture(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(150):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(2, 7)), int(rng.integers(0, 256)), -1)
    for _ in range(80):
        x, y = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        s = int(rng.integers(5, 18))
        cv2.rectangle(img, (x, y), (x + s, y + s), int(rng.integers(0, 256)), -1)
    return img


def _plane_homography(depth_z: float) -> np.ndarray:
    return K_B @ (R_AB - np.outer(T_AB, [0.0, 0.0, 1.0]) / depth_z) @ np.linalg.inv(K_A)


def _multidepth_pair(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    depths = [5.0, 8.0, 13.0]
    tex_seeds = [seed + 1, seed + 3, seed + 5]
    img_a = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    img_b = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for depth, tseed in zip(depths, tex_seeds):
        tex = _texture(tseed)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _ in range(25):
            x = int(rng.integers(0, IMG_W - 120))
            y = int(rng.integers(0, IMG_H - 120))
            cv2.rectangle(
                mask,
                (x, y),
                (x + int(rng.integers(60, 160)), y + int(rng.integers(60, 160))),
                255,
                -1,
            )
        Hab = _plane_homography(depth)
        tex_b = cv2.warpPerspective(tex, Hab, (IMG_W, IMG_H))
        mask_b = cv2.warpPerspective(mask, Hab, (IMG_W, IMG_H))
        img_a[mask > 127] = tex[mask > 127]
        img_b[mask_b > 127] = tex_b[mask_b > 127]
    return img_a, img_b


def _degrade(img: np.ndarray, seed: int) -> np.ndarray:
    """Crude low-contrast + noise + blur, vaguely radiograph-like."""
    lowc = (img.astype(np.float32) * 0.35 + 90.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, size=img.shape).astype(np.float32)
    blurred = cv2.GaussianBlur((lowc + noise).clip(0, 255), (3, 3), 0.7)
    return blurred.astype(np.uint8)


def test_analyze_pair_returns_expected_structure_per_variant():
    a, b = _multidepth_pair()
    results = analyze_pair(a, b)

    assert isinstance(results, list)
    assert {r.variant for r in results} == set(VARIANTS)
    for r in results:
        assert isinstance(r, VariantResult)
        assert isinstance(r.kp_a, int) and r.kp_a >= 0
        assert isinstance(r.kp_b, int) and r.kp_b >= 0
        assert isinstance(r.good_matches, int) and r.good_matches >= 0
        assert isinstance(r.success, bool)
        if r.success:
            assert isinstance(r.inlier_count, int) and r.inlier_count >= 8
            assert isinstance(r.mean_epipolar_error, float)
            assert math.isfinite(r.mean_epipolar_error)
            assert 0 <= r.coverage_a <= 16
            assert 0 <= r.coverage_b <= 16
        else:
            assert isinstance(r.failure_reason, str) and r.failure_reason


def test_clahe_actually_changes_the_image():
    a, _ = _multidepth_pair()
    degraded = _degrade(a, seed=1)
    clahe_out = preprocess(degraded, "clahe")
    eq_out = preprocess(degraded, "equalize")
    raw_out = preprocess(degraded, "raw")

    assert clahe_out.shape == degraded.shape
    assert clahe_out.dtype == np.uint8
    assert not np.array_equal(clahe_out, degraded), "CLAHE produced identical pixels"
    assert not np.array_equal(eq_out, degraded), "equalize produced identical pixels"
    assert np.array_equal(raw_out, degraded), "raw must be a passthrough"


def test_grid_coverage_returns_value_in_zero_to_sixteen():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, size=(500, 2)).astype(np.float32)
    cov = grid_coverage(pts, h=480, w=640)
    assert isinstance(cov, int)
    assert 0 <= cov <= 16

    # corner points only -> exactly the 4 corner cells
    corners = np.float32([[0, 0], [639, 0], [0, 479], [639, 479]])
    assert grid_coverage(corners, h=480, w=640) == 4

    # empty -> 0
    empty = np.empty((0, 2), dtype=np.float32)
    assert grid_coverage(empty, h=480, w=640) == 0


def test_preprocess_rejects_bad_variant():
    a, _ = _multidepth_pair()
    try:
        preprocess(a, "not_a_variant")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown variant")


def test_analyze_pair_records_failure_when_match_count_below_floor():
    # Two unrelated uniform-ish images: SIFT should find very few stable
    # matches, exercising the "good_matches < 8" branch on at least one
    # variant. We don't assert which variant — just that failure is recorded
    # cleanly with a reason and no exception.
    flat_a = np.full((IMG_H, IMG_W), 100, dtype=np.uint8)
    flat_a[0, 0] = 101
    flat_b = np.full((IMG_H, IMG_W), 100, dtype=np.uint8)
    flat_b[1, 1] = 101

    results = analyze_pair(flat_a, flat_b)
    assert all(not r.success for r in results)
    assert all(r.failure_reason for r in results)
