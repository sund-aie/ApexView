import math

import cv2
import numpy as np
import pytest

from apexview.engine.extension_stitch import (
    InsufficientOverlapError,
    StitchResult,
    stitch_extension,
)

SEED = 1234
IMG_H = 240
IMG_W = 320


def _make_feature_image(seed: int = SEED, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Procedural high-contrast grayscale image SIFT can latch onto.

    Mixes a coarse checkerboard with scattered bright/dark blobs at
    pseudorandom positions. Fully deterministic for the given seed.
    """
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


def _build_extension_homography(
    dx: float, dy: float, angle_deg: float, center: tuple[float, float]
) -> np.ndarray:
    """Build a 3x3 homography from a small rotation about a center + translation."""
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


def test_known_homography_is_recovered_within_one_pixel():
    image_a = _make_feature_image()

    dx, dy, angle = 60.0, 4.0, 2.0
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    true_h = _build_extension_homography(dx, dy, angle, (cx, cy))

    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    result = stitch_extension(image_a, image_b)

    assert result.mean_reprojection_error <= 1.0, (
        f"reprojection error too high: {result.mean_reprojection_error}"
    )
    assert result.inlier_count >= 20, (
        f"too few inliers for a clean synthetic case: {result.inlier_count}"
    )

    expected_h = np.linalg.inv(true_h)
    corners = np.float32(
        [[0, 0], [0, IMG_H], [IMG_W, IMG_H], [IMG_W, 0]]
    ).reshape(-1, 1, 2)
    recovered_corners = cv2.perspectiveTransform(corners, result.homography)
    expected_corners = cv2.perspectiveTransform(corners, expected_h)
    corner_errors = np.linalg.norm(
        recovered_corners - expected_corners, axis=2
    ).ravel()
    max_corner_error = float(corner_errors.max())
    assert max_corner_error <= 1.0, (
        f"recovered homography deviates from ground truth: "
        f"max corner error {max_corner_error:.4f} px"
    )


def test_unmatchable_features_raise_insufficient_overlap():
    image_a = _make_feature_image(seed=11)
    rng = np.random.default_rng(424242)
    image_b = rng.integers(0, 256, size=(IMG_H, IMG_W), dtype=np.uint8)

    with pytest.raises(InsufficientOverlapError):
        stitch_extension(image_a, image_b)


def test_blank_image_raises_insufficient_overlap():
    image_a = _make_feature_image()
    blank = np.full_like(image_a, 200)
    with pytest.raises(InsufficientOverlapError):
        stitch_extension(image_a, blank)


def test_rejects_non_2d_array():
    image_a = _make_feature_image()
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        stitch_extension(image_a, bad)


def test_rejects_empty_array():
    image_a = _make_feature_image()
    empty = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        stitch_extension(image_a, empty)


def test_rejects_mismatched_dtypes():
    image_a = _make_feature_image()
    image_b = image_a.astype(np.uint16)
    with pytest.raises(ValueError):
        stitch_extension(image_a, image_b)


def test_output_contract():
    image_a = _make_feature_image()
    true_h = _build_extension_homography(50.0, 3.0, 1.5, (IMG_W / 2.0, IMG_H / 2.0))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    result = stitch_extension(image_a, image_b)

    assert isinstance(result, StitchResult)
    assert isinstance(result.stitched_image, np.ndarray)
    assert isinstance(result.homography, np.ndarray)
    assert isinstance(result.inlier_count, int)
    assert isinstance(result.mean_reprojection_error, float)
    assert math.isfinite(result.mean_reprojection_error)

    assert result.homography.shape == (3, 3)
    assert result.stitched_image.ndim == 2

    h_out, w_out = result.stitched_image.shape
    assert h_out >= IMG_H
    assert w_out >= IMG_W
    assert (h_out * w_out) > (IMG_H * IMG_W), (
        "stitched canvas should be strictly larger than a single input (no clipping)"
    )


def test_apply_clahe_false_bypasses_preprocessing():
    """With apply_clahe=False, the engine must reproduce the original
    raw-input behavior: the known-homography recovery still holds to the
    same tight tolerance, AND the False branch must produce a measurably
    different inlier set than the True branch (otherwise the toggle is
    being silently ignored)."""
    image_a = _make_feature_image()
    dx, dy, angle = 60.0, 4.0, 2.0
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    true_h = _build_extension_homography(dx, dy, angle, (cx, cy))
    image_b = cv2.warpPerspective(image_a, true_h, (IMG_W, IMG_H))

    raw_result = stitch_extension(image_a, image_b, apply_clahe=False)
    clahe_result = stitch_extension(image_a, image_b, apply_clahe=True)

    assert raw_result.mean_reprojection_error <= 1.0, (
        f"raw-mode reprojection error too high: {raw_result.mean_reprojection_error}"
    )
    expected_h = np.linalg.inv(true_h)
    corners = np.float32(
        [[0, 0], [0, IMG_H], [IMG_W, IMG_H], [IMG_W, 0]]
    ).reshape(-1, 1, 2)
    max_corner_error = float(
        np.linalg.norm(
            cv2.perspectiveTransform(corners, raw_result.homography)
            - cv2.perspectiveTransform(corners, expected_h),
            axis=2,
        ).max()
    )
    assert max_corner_error <= 1.0

    # If the toggle were ignored, both calls would produce identical results.
    assert raw_result.inlier_count != clahe_result.inlier_count, (
        "apply_clahe toggle appears to have no effect on matching results; "
        "the parameter may not be threaded into preprocess_for_matching"
    )


def test_clahe_does_not_mutate_caller_images():
    image_a = _make_feature_image()
    image_b = cv2.warpPerspective(
        image_a, _build_extension_homography(60.0, 4.0, 2.0, (IMG_W / 2.0, IMG_H / 2.0)),
        (IMG_W, IMG_H),
    )
    snap_a = image_a.copy()
    snap_b = image_b.copy()
    _ = stitch_extension(image_a, image_b)  # default apply_clahe=True
    assert np.array_equal(image_a, snap_a)
    assert np.array_equal(image_b, snap_b)
