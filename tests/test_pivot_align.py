"""Tests for the rough overlap aligner used by the pivot viewer.

Synthetic and deterministic. Uses the same procedural-texture recipe as
the extension-stitch tests so SIFT has something to latch onto.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from apexview.engine.pivot_align import (
    PivotAlignment,
    PivotAlignmentError,
    align_for_pivot,
    validate_alignment_homography,
)

SEED = 1234
IMG_H = 240
IMG_W = 320


def _make_feature_image(seed: int = SEED, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)

    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 40
            else:
                img[y : y + tile, x : x + tile] = 215

    rng = np.random.default_rng(seed)
    for _ in range(80):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        radius = int(rng.integers(3, 8))
        color = int(rng.integers(0, 256))
        cv2.circle(img, (cx, cy), radius, color, thickness=-1)

    for _ in range(40):
        x1 = int(rng.integers(5, w - 25))
        y1 = int(rng.integers(5, h - 25))
        side = int(rng.integers(6, 18))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color, thickness=-1)

    return img


def _mild_homography(
    dx: float, dy: float, angle_deg: float, center: tuple[float, float]
) -> np.ndarray:
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


def _central_mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference over the central 50% of the frame.

    Margins are excluded because the warp pulls in zero-fill at the edges,
    which would dominate the metric and hide the actual overlap quality.
    """
    h, w = a.shape
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    diff = np.abs(a[y0:y1, x0:x1].astype(np.float32) - b[y0:y1, x0:x1].astype(np.float32))
    return float(diff.mean())


def test_returns_warped_image_close_to_reference_on_overlap():
    image_a = _make_feature_image()
    true_h = _mild_homography(40.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    alignment = align_for_pivot(image_a, image_b)

    assert isinstance(alignment, PivotAlignment)
    assert alignment.warped_b_to_a.shape == image_a.shape
    assert alignment.warped_b_to_a.dtype == np.uint8
    assert alignment.image_a is image_a
    assert alignment.homography.shape == (3, 3)
    assert isinstance(alignment.inlier_count, int)
    assert alignment.inlier_count >= 20, (
        f"too few inliers for a clean synthetic case: {alignment.inlier_count}"
    )
    assert isinstance(alignment.mean_alignment_error_px, float)
    assert math.isfinite(alignment.mean_alignment_error_px)
    assert alignment.mean_alignment_error_px <= 3.0, (
        f"alignment error unexpectedly high: {alignment.mean_alignment_error_px}"
    )

    central_diff = _central_mean_abs_diff(alignment.warped_b_to_a, image_a)
    assert central_diff < 25.0, (
        f"central-region mean abs diff after warp too high: {central_diff}"
    )


def test_warp_actually_reduces_misalignment_vs_unwarped_b():
    """Bite-style proof: the returned homography must actually pull image_b
    closer to image_a than the unwarped image_b would be. A no-op identity
    alignment would fail this assertion."""
    image_a = _make_feature_image()
    true_h = _mild_homography(40.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    alignment = align_for_pivot(image_a, image_b)

    unwarped_diff = _central_mean_abs_diff(image_b, image_a)
    warped_diff = _central_mean_abs_diff(alignment.warped_b_to_a, image_a)

    assert warped_diff < unwarped_diff, (
        f"alignment did not reduce misalignment: "
        f"warped={warped_diff}, unwarped={unwarped_diff}"
    )


def test_unrelated_noise_images_raise_pivot_alignment_error():
    image_a = _make_feature_image(seed=11)
    rng = np.random.default_rng(424242)
    image_b = rng.integers(0, 256, size=(IMG_H, IMG_W), dtype=np.uint8)

    with pytest.raises(PivotAlignmentError):
        align_for_pivot(image_a, image_b)


def test_blank_image_raises_pivot_alignment_error():
    image_a = _make_feature_image()
    blank = np.full_like(image_a, 200)
    with pytest.raises(PivotAlignmentError):
        align_for_pivot(image_a, blank)


def test_rejects_non_2d_array():
    image_a = _make_feature_image()
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        align_for_pivot(image_a, bad)


def test_rejects_empty_array():
    image_a = _make_feature_image()
    empty = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        align_for_pivot(image_a, empty)


def test_rejects_mismatched_dtypes():
    image_a = _make_feature_image()
    image_b = image_a.astype(np.uint16)
    with pytest.raises(ValueError):
        align_for_pivot(image_a, image_b)


def test_rejects_non_array_input():
    image_a = _make_feature_image()
    with pytest.raises(ValueError):
        align_for_pivot(image_a, "not an array")  # type: ignore[arg-type]


def test_alignment_is_frozen_dataclass():
    image_a = _make_feature_image()
    image_b = cv2.warpPerspective(
        image_a,
        _mild_homography(30.0, 2.0, 1.0, (IMG_W / 2.0, IMG_H / 2.0)),
        (IMG_W, IMG_H),
    )
    alignment = align_for_pivot(image_a, image_b)
    with pytest.raises(Exception):
        alignment.inlier_count = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_alignment_homography — known-answer on the real-world failure
# ---------------------------------------------------------------------------

# The exact degenerate homography recovered from a real angulation pair:
# 16 Lowe matches, only 5 RANSAC inliers, and a fitted matrix whose line at
# infinity crosses the image (corner homogeneous w values
# [1.0, -1.89, -2.85, 0.05] — a sign change across the frame), so
# warpPerspective folded the whole radiograph through a point into a "fan".
# The source image shape was (356, 481).
H_FAN = np.array([
    [-1.73769461e+00, -7.64199657e-01,  2.87719376e+02],
    [-8.00036351e-01, -3.78610331e-01,  1.35733853e+02],
    [-6.01600323e-03, -2.68035253e-03,  1.00000000e+00],
])
H_FAN_SHAPE = (356, 481)


def test_validator_rejects_real_world_fan_matrix():
    ok, reason = validate_alignment_homography(H_FAN, H_FAN_SHAPE)
    assert ok is False
    lowered = reason.lower()
    assert (
        "corner" in lowered or "infinity" in lowered or "fold" in lowered
    ), f"reason must reference the fold/infinity/corner check, got: {reason}"


def test_validator_accepts_identity():
    ok, reason = validate_alignment_homography(np.eye(3), (IMG_H, IMG_W))
    assert (ok, reason) == (True, "")


def test_validator_accepts_mild_rotation_translation():
    h = _mild_homography(30.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    ok, reason = validate_alignment_homography(h, (IMG_H, IMG_W))
    assert (ok, reason) == (True, "")


def test_validator_rejects_reflection():
    reflect = np.diag([-1.0, 1.0, 1.0])
    ok, reason = validate_alignment_homography(reflect, (IMG_H, IMG_W))
    assert ok is False
    lowered = reason.lower()
    assert "reflect" in lowered or "winding" in lowered or "convex" in lowered


def test_validator_rejects_extreme_scale():
    tiny = np.diag([0.01, 0.01, 1.0])
    ok, reason = validate_alignment_homography(tiny, (IMG_H, IMG_W))
    assert ok is False
    assert "area" in reason.lower()


def test_validator_rejects_zero_normalization_term():
    h = np.eye(3)
    h[2, 2] = 0.0
    ok, reason = validate_alignment_homography(h, (IMG_H, IMG_W))
    assert ok is False
    assert reason  # a real explanation, not an empty string


def test_validator_rejects_nan():
    h = np.eye(3)
    h[0, 1] = np.nan
    ok, reason = validate_alignment_homography(h, (IMG_H, IMG_W))
    assert ok is False
    assert "finite" in reason.lower() or "nan" in reason.lower()


def test_validator_rejects_wrong_shape_matrix():
    ok, reason = validate_alignment_homography(
        np.eye(2), (IMG_H, IMG_W)
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Trust floor and validator wiring inside align_for_pivot (deterministic
# via monkeypatched cv2.findHomography; two identical textured images feed
# SIFT plenty of matches so the pipeline reliably reaches RANSAC)
# ---------------------------------------------------------------------------


def test_low_inlier_count_is_refused_even_with_sane_homography(monkeypatch):
    """5 RANSAC inliers (the real-world garbage case) must be refused even
    when the fitted matrix itself looks sane: counts near the 4-point
    existence minimum carry no evidence the alignment is real."""
    import apexview.engine.pivot_align as pa

    image_a = _make_feature_image()
    image_b = image_a.copy()
    sane = _mild_homography(10.0, 2.0, 1.0, (IMG_W / 2.0, IMG_H / 2.0))

    calls: list[int] = []

    def fake_find_homography(src_pts, dst_pts, method, threshold):
        calls.append(src_pts.shape[0])
        mask = np.zeros((src_pts.shape[0], 1), dtype=np.uint8)
        mask[:5] = 1  # exactly 5 inliers out of N
        return sane, mask

    monkeypatch.setattr(pa.cv2, "findHomography", fake_find_homography)

    with pytest.raises(PivotAlignmentError) as excinfo:
        align_for_pivot(image_a, image_b)

    assert calls, "findHomography was never reached; pre-fit gate fired instead"
    message = str(excinfo.value).lower()
    assert "too few" in message
    assert "5" in str(excinfo.value)


def test_degenerate_homography_with_enough_inliers_is_refused(monkeypatch):
    """The validator must be wired into align_for_pivot, not just exist:
    feed RANSAC output that has plenty of inliers but the real-world fan
    matrix, and the alignment must still be refused."""
    import apexview.engine.pivot_align as pa

    image_a = _make_feature_image()
    image_b = image_a.copy()

    calls: list[int] = []

    def fake_find_homography(src_pts, dst_pts, method, threshold):
        calls.append(src_pts.shape[0])
        mask = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return H_FAN.copy(), mask

    monkeypatch.setattr(pa.cv2, "findHomography", fake_find_homography)

    with pytest.raises(PivotAlignmentError) as excinfo:
        align_for_pivot(image_a, image_b)

    assert calls, "findHomography was never reached; pre-fit gate fired instead"
    assert "trustworthy" in str(excinfo.value).lower()
