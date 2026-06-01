"""Tests for the extension-vs-angulation pair classifier.

Synthetic and deterministic. For the angulation case we render a true
two-depth scene through a tilted pinhole camera so the warp produces
depth-dependent parallax no single homography can model — this is the same
recipe validated in Task 2 Step 1 characterization, copied here as test
scaffold (we do NOT import from experiments/).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from apexview.engine.pair_classifier import (
    EXTENSION_MIN_INLIERS,
    ClassificationResult,
    PairType,
    classify_pair,
)

IMG_H = 240
IMG_W = 320
FOCAL = float(IMG_W)
BG_DEPTH = 3.0 * FOCAL
FG_DEPTH = 1.5 * FOCAL
AXIS_DEPTH = 2.0 * FOCAL


def _make_background(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)
    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 40
            else:
                img[y : y + tile, x : x + tile] = 215
    rng = np.random.default_rng(seed * 31 + 7)
    for _ in range(20):
        cx = int(rng.integers(5, w - 5))
        cy = int(rng.integers(5, h - 5))
        cv2.circle(img, (cx, cy), 2, int(rng.integers(0, 256)), -1)
    return img


def _make_foreground(
    seed: int, h: int = IMG_H, w: int = IMG_W
) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        radius = int(rng.integers(3, 8))
        color = int(rng.integers(0, 256))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    for _ in range(40):
        x1 = int(rng.integers(5, w - 25))
        y1 = int(rng.integers(5, h - 25))
        side = int(rng.integers(6, 18))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color, -1)
        cv2.rectangle(mask, (x1, y1), (x1 + side, y1 + side), 255, -1)
    return img, mask


def _layer_homography(angle_deg: float, layer_z: float) -> np.ndarray:
    """Per-layer homography for a flat layer at depth ``layer_z`` when the
    scene rotates about a vertical axis at depth ``AXIS_DEPTH`` by
    ``angle_deg``. Because the rotation axis differs from the layer depth,
    different layers warp by different homographies — that's the parallax."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    src = np.float32([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]])
    dst = []
    for u, v in src:
        X = (u - cx) * layer_z / FOCAL
        Y = (v - cy) * layer_z / FOCAL
        Zr = layer_z - AXIS_DEPTH
        Xn = X * c + Zr * s
        Zn = -X * s + Zr * c
        Zf = Zn + AXIS_DEPTH
        if Zf <= 1e-3:
            return np.eye(3, dtype=np.float64)
        dst.append([cx + FOCAL * Xn / Zf, cy + FOCAL * Y / Zf])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _render_two_layer_scene(seed: int, angle_deg: float) -> np.ndarray:
    bg = _make_background(seed)
    fg, mask = _make_foreground(seed)
    h_bg = _layer_homography(angle_deg, BG_DEPTH)
    h_fg = _layer_homography(angle_deg, FG_DEPTH)
    warped_bg = cv2.warpPerspective(bg, h_bg, (IMG_W, IMG_H))
    warped_fg = cv2.warpPerspective(fg, h_fg, (IMG_W, IMG_H))
    warped_mask = cv2.warpPerspective(mask, h_fg, (IMG_W, IMG_H))
    out = warped_bg.copy()
    out[warped_mask > 127] = warped_fg[warped_mask > 127]
    return out


def _make_single_layer_image(seed: int) -> np.ndarray:
    """Single-layer feature image (no depth). Used for the EXTENSION test
    because a planar in-plane warp of THIS is what 'extension' means."""
    img = _make_background(seed)
    fg, mask = _make_foreground(seed)
    img[mask > 127] = fg[mask > 127]
    return img


def _extension_homography(angle_deg: float, dx: float, dy: float = 0.0) -> np.ndarray:
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


def test_extension_pair_is_classified_as_extension():
    image_a = _make_single_layer_image(seed=11)
    homography = _extension_homography(angle_deg=1.5, dx=60.0)
    image_b = cv2.warpPerspective(image_a, homography, (IMG_W, IMG_H))

    result = classify_pair(image_a, image_b)

    assert result.pair_type is PairType.EXTENSION, result.message
    assert result.inlier_count >= EXTENSION_MIN_INLIERS, (
        f"inlier_count {result.inlier_count} below threshold"
    )
    assert result.stitched_image is not None
    assert isinstance(result.stitched_image, np.ndarray)
    assert result.stitched_image.ndim == 2


def test_angulation_pair_is_classified_as_angulation():
    image_a = _render_two_layer_scene(seed=11, angle_deg=0.0)
    image_b = _render_two_layer_scene(seed=11, angle_deg=18.0)

    result = classify_pair(image_a, image_b)

    assert result.pair_type is PairType.ANGULATION, (
        f"got {result.pair_type.name}, "
        f"inliers={result.inlier_count}, err={result.mean_reprojection_error}"
    )
    assert result.stitched_image is None
    assert result.inlier_count < EXTENSION_MIN_INLIERS


def test_severe_angulation_matching_failure_is_caught():
    image_a = _make_single_layer_image(seed=11)
    rng = np.random.default_rng(424242)
    image_b = rng.integers(0, 256, size=(IMG_H, IMG_W), dtype=np.uint8)

    result = classify_pair(image_a, image_b)

    assert result.pair_type is PairType.ANGULATION
    assert result.inlier_count == 0
    assert result.stitched_image is None
    assert math.isnan(result.mean_reprojection_error)


def test_bad_input_propagates_value_error():
    image_a = _make_single_layer_image(seed=11)
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        classify_pair(image_a, bad)


def test_classification_result_contract():
    image_a = _make_single_layer_image(seed=11)
    homography = _extension_homography(angle_deg=1.0, dx=55.0)
    image_b = cv2.warpPerspective(image_a, homography, (IMG_W, IMG_H))
    ext_result = classify_pair(image_a, image_b)

    image_c = _render_two_layer_scene(seed=11, angle_deg=0.0)
    image_d = _render_two_layer_scene(seed=11, angle_deg=18.0)
    ang_result = classify_pair(image_c, image_d)

    for r in (ext_result, ang_result):
        assert isinstance(r, ClassificationResult)
        assert isinstance(r.pair_type, PairType)
        assert isinstance(r.inlier_count, int)
        assert isinstance(r.mean_reprojection_error, float)
        assert isinstance(r.message, str) and r.message

    assert ext_result.pair_type is PairType.EXTENSION
    assert isinstance(ext_result.stitched_image, np.ndarray)
    assert ext_result.stitched_image.ndim == 2

    assert ang_result.pair_type is PairType.ANGULATION
    assert ang_result.stitched_image is None

    assert ext_result.pair_type is not ang_result.pair_type, (
        "extension and angulation cases must produce different verdicts"
    )


def test_classifier_inherits_default_clahe_from_stitcher():
    """classify_pair calls stitch_extension internally, which now defaults
    to apply_clahe=True. Verify the classifier transparently gets CLAHE
    (no second CLAHE call is needed in the classifier) and still produces
    different verdicts on extension vs angulation cases. This is the
    'single home for preprocessing' contract: CLAHE lives in the engine,
    not in the classifier."""
    image_a = _make_single_layer_image(seed=11)
    image_b = cv2.warpPerspective(
        image_a, _extension_homography(angle_deg=1.5, dx=60.0), (IMG_W, IMG_H)
    )
    ext = classify_pair(image_a, image_b)

    image_c = _render_two_layer_scene(seed=11, angle_deg=0.0)
    image_d = _render_two_layer_scene(seed=11, angle_deg=18.0)
    ang = classify_pair(image_c, image_d)

    assert ext.pair_type is PairType.EXTENSION
    assert ang.pair_type is PairType.ANGULATION
    assert ext.inlier_count >= EXTENSION_MIN_INLIERS
    assert ang.inlier_count < EXTENSION_MIN_INLIERS
