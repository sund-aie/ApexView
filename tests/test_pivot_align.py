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


def _overlap_mean_abs_diff(canvas_a: np.ndarray, canvas_b: np.ndarray) -> float:
    """Mean absolute pixel difference where BOTH canvases carry content.

    Zero pixels mark regions a radiograph does not cover on the union
    canvas, so the (a > 0) & (b > 0) mask restricts the metric to the
    true overlap; including either zero-fill region would dominate the
    metric and hide the actual registration quality.
    """
    overlap = (canvas_a > 0) & (canvas_b > 0)
    assert overlap.any(), "canvases share no overlap at all"
    diff = np.abs(
        canvas_a.astype(np.float32) - canvas_b.astype(np.float32)
    )
    return float(diff[overlap].mean())


def test_union_canvas_contract_on_mild_warp():
    image_a = _make_feature_image()
    true_h = _mild_homography(40.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    alignment = align_for_pivot(image_a, image_b)

    assert isinstance(alignment, PivotAlignment)
    assert alignment.canvas_a.shape == alignment.canvas_b.shape
    assert alignment.canvas_a.dtype == np.uint8
    assert alignment.canvas_b.dtype == np.uint8
    canvas_h, canvas_w = alignment.canvas_a.shape
    assert canvas_h >= IMG_H and canvas_w >= IMG_W, (
        "union canvas must be at least as large as image_a"
    )
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

    # image_a is placed by exact slice assignment, never resampled.
    x_min, y_min = alignment.canvas_offset
    assert x_min <= 0 and y_min <= 0, (
        "union box must contain image_a's own origin"
    )
    oy, ox = -y_min, -x_min
    assert np.array_equal(
        alignment.canvas_a[oy : oy + IMG_H, ox : ox + IMG_W], image_a
    ), "canvas_a at the placement offset must equal image_a exactly"

    # Everywhere outside the placement, canvas_a is untouched zero.
    placed = np.zeros((canvas_h, canvas_w), dtype=bool)
    placed[oy : oy + IMG_H, ox : ox + IMG_W] = True
    assert np.all(alignment.canvas_a[~placed] == 0)


def test_registration_quality_on_overlap():
    """Where both canvases carry content they must agree closely — the
    same strength as the old central-region check (threshold 25), now
    measured over the real overlap mask instead of a fixed crop."""
    image_a = _make_feature_image()
    true_h = _mild_homography(40.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    alignment = align_for_pivot(image_a, image_b)

    overlap = (alignment.canvas_a > 0) & (alignment.canvas_b > 0)
    assert overlap.sum() >= 0.30 * IMG_H * IMG_W, (
        f"overlap region too small to be meaningful: {overlap.sum()} px"
    )
    overlap_diff = _overlap_mean_abs_diff(alignment.canvas_a, alignment.canvas_b)
    assert overlap_diff < 25.0, (
        f"overlap mean abs diff after warp too high: {overlap_diff}"
    )


def test_translated_pair_grows_canvas_and_b_extends_view():
    """THE user-requested behavior, known answer: crop two overlapping
    windows from one wide scene, 60 px apart. The union canvas must grow
    sideways to hold both, and the region beyond image_a's width must
    carry real content from B — the old clipped warp at image_a's (H, W)
    discarded exactly that content."""
    scene = _make_feature_image(seed=SEED, h=IMG_H, w=IMG_W + 60)
    image_a = scene[:, :IMG_W].copy()
    image_b = scene[:, 60 : 60 + IMG_W].copy()

    alignment = align_for_pivot(image_a, image_b)

    canvas_h, canvas_w = alignment.canvas_a.shape
    assert canvas_w >= IMG_W + 40, (
        f"canvas did not grow sideways to hold both: width {canvas_w}"
    )

    # In canvas coordinates image_a ends at column ox + IMG_W; beyond it
    # only B can supply content, and it must actually do so.
    x_min, y_min = alignment.canvas_offset
    ox = -x_min
    beyond_b = alignment.canvas_b[:, ox + IMG_W :]
    assert beyond_b.size > 0
    assert np.count_nonzero(beyond_b) > 0.5 * beyond_b.size, (
        "canvas_b carries no real content beyond image_a's right edge"
    )
    # And image_a itself is fully present on its canvas (nothing cropped).
    oy = -y_min
    assert np.array_equal(
        alignment.canvas_a[oy : oy + IMG_H, ox : ox + IMG_W], image_a
    )


def test_alignment_beats_naive_unwarped_placement():
    """Bite-style proof: the warped canvas_b must agree with canvas_a
    strictly better than naively dropping the unwarped image_b at the
    same offset. An identity/no-op alignment fails this assertion."""
    image_a = _make_feature_image()
    true_h = _mild_homography(40.0, 5.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    alignment = align_for_pivot(image_a, image_b)

    x_min, y_min = alignment.canvas_offset
    oy, ox = -y_min, -x_min
    canvas_b_naive = np.zeros_like(alignment.canvas_a)
    canvas_b_naive[oy : oy + IMG_H, ox : ox + IMG_W] = image_b

    aligned_diff = _overlap_mean_abs_diff(alignment.canvas_a, alignment.canvas_b)
    naive_diff = _overlap_mean_abs_diff(alignment.canvas_a, canvas_b_naive)

    assert aligned_diff < naive_diff, (
        f"alignment did not beat naive unwarped placement: "
        f"aligned={aligned_diff}, naive={naive_diff}"
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
    with pytest.raises(Exception):
        alignment.canvas_a = None  # type: ignore[misc]
    with pytest.raises(Exception):
        alignment.canvas_offset = (0, 0)  # type: ignore[misc]


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


def test_huge_translation_canvas_is_refused(monkeypatch):
    """Canvas blow-up guard, wired AFTER the validator: a pure translation
    of 10000 px passes every validator check (corner w stays 1, winding
    kept, area ratio exactly 1) yet the union canvas would be ~32x the
    input area — almost entirely black. The guard must refuse it with the
    barely-overlap message."""
    import apexview.engine.pivot_align as pa

    image_a = _make_feature_image()
    image_b = image_a.copy()

    huge_shift = np.eye(3, dtype=np.float64)
    huge_shift[0, 2] = 10000.0
    # Sanity precondition: this matrix is validator-clean, so a refusal
    # can only come from the canvas guard downstream of the validator.
    ok, reason = validate_alignment_homography(huge_shift, (IMG_H, IMG_W))
    assert (ok, reason) == (True, "")

    calls: list[int] = []

    def fake_find_homography(src_pts, dst_pts, method, threshold):
        calls.append(src_pts.shape[0])
        mask = np.ones((src_pts.shape[0], 1), dtype=np.uint8)
        return huge_shift.copy(), mask

    monkeypatch.setattr(pa.cv2, "findHomography", fake_find_homography)

    with pytest.raises(PivotAlignmentError) as excinfo:
        align_for_pivot(image_a, image_b)

    assert calls, "findHomography was never reached; pre-fit gate fired instead"
    message = str(excinfo.value).lower()
    assert "combined view" in message or "overlap" in message
