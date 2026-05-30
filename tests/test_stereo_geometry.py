"""Tests for two-view fundamental-matrix estimation.

Fully synthetic and deterministic. Two layers of verification, kept separate
so a math bug can be told apart from a SIFT-matching limitation:

  * EXACT-MATH tests project known multi-depth 3D points through two cameras
    with known intrinsics K and known relative pose (R, t), giving exact
    correspondences and an analytically-constructed ground-truth F. These pin
    the F math and the epipolar-direction convention to tight tolerance.
  * THROUGH-SIFT test renders a multi-depth procedural image pair and runs the
    full image -> SIFT -> F pipeline, tolerating SIFT localization noise.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from apexview.engine.stereo_geometry import (
    InsufficientGeometryError,
    TwoViewGeometry,
    estimate_fundamental_from_points,
    estimate_two_view_geometry,
    mean_symmetric_epipolar_error,
)

IMG_H = 480
IMG_W = 640

# Known intrinsics for the two cameras.
K_A = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
K_B = np.array([[820.0, 0.0, 310.0], [0.0, 820.0, 250.0], [0.0, 0.0, 1.0]])

# Known relative pose: small rotation about the vertical axis + a translation
# with a dominant sideways (baseline) component.
_THETA = math.radians(8.0)
R_AB = np.array(
    [
        [math.cos(_THETA), 0.0, math.sin(_THETA)],
        [0.0, 1.0, 0.0],
        [-math.sin(_THETA), 0.0, math.cos(_THETA)],
    ]
)
T_AB = np.array([0.6, 0.05, 0.1])


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]
    )


def _ground_truth_F() -> np.ndarray:
    """Analytic F = K_b^{-T} [t]_x R K_a^{-1}, normalized to unit Frobenius."""
    F = np.linalg.inv(K_B).T @ _skew(T_AB) @ R_AB @ np.linalg.inv(K_A)
    return F / np.linalg.norm(F)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D points (rows) given camera K[R|t]; returns (N,2) pixels."""
    Xc = (R @ X.T).T + t
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


def _multidepth_points(seed: int = 0, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Exact correspondences from points spanning a range of depths."""
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, n),
            rng.uniform(-1.5, 1.5, n),
            rng.uniform(4.0, 12.0, n),  # varied depth -> non-degenerate F
        ]
    )
    pts_a = _project(K_A, np.eye(3), np.zeros(3), X)
    pts_b = _project(K_B, R_AB, T_AB, X)
    return pts_a, pts_b


def _planar_points(seed: int = 1, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Exact correspondences from a single 3D plane (degenerate for F)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = 3.0 * x + 2.0 * y + 8.0  # all points satisfy one plane equation
    X = np.column_stack([x, y, z])
    pts_a = _project(K_A, np.eye(3), np.zeros(3), X)
    pts_b = _project(K_B, R_AB, T_AB, X)
    return pts_a, pts_b


def _normalize_sign(F: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Unit-Frobenius normalize F and align its sign with a reference."""
    Fn = F / np.linalg.norm(F)
    if np.vdot(Fn, reference) < 0:
        Fn = -Fn
    return Fn


# --------------------------------------------------------------------------
# Multi-depth procedural image pair (for the through-SIFT test)
# --------------------------------------------------------------------------
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
    """A->B inter-view homography for a fronto-parallel plane at ``depth_z``."""
    return K_B @ (R_AB - np.outer(T_AB, [0.0, 0.0, 1.0]) / depth_z) @ np.linalg.inv(K_A)


def _multidepth_image_pair(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """Render a 3-layer scene from camera A (reference) and camera B.

    Each layer is a fronto-parallel textured plane at a distinct depth, warped
    into camera B by its plane-induced homography. The depth spread produces
    genuine parallax between layers, so no single homography (and hence no
    planar degeneracy) explains the whole pair.
    """
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


# --------------------------------------------------------------------------
# Test 1: known-geometry math (exact points, no SIFT)
# --------------------------------------------------------------------------
def test_known_geometry_math_recovers_F():
    pts_a, pts_b = _multidepth_points()
    F_gt = _ground_truth_F()

    F_est, mask = estimate_fundamental_from_points(pts_a, pts_b)

    # Epipolar constraint holds on exact input to tight tolerance.
    err = mean_symmetric_epipolar_error(F_est, pts_a[mask], pts_b[mask])
    assert err <= 1e-3, f"epipolar error on exact points too high: {err}"

    # Recovered F agrees with the analytic ground truth (up to scale + sign).
    F_est_n = _normalize_sign(F_est, F_gt)
    max_diff = np.abs(F_est_n - F_gt).max()
    assert max_diff <= 1e-2, (
        f"recovered F disagrees with analytic ground truth: max element "
        f"difference {max_diff}"
    )


# --------------------------------------------------------------------------
# Test 2: end-to-end through SIFT
# --------------------------------------------------------------------------
def test_end_to_end_through_sift():
    img_a, img_b = _multidepth_image_pair()

    geom = estimate_two_view_geometry(img_a, img_b)

    assert isinstance(geom, TwoViewGeometry)
    assert geom.inlier_count >= 8
    assert geom.mean_epipolar_error <= 2.0, (
        f"SIFT-pipeline epipolar error too high: {geom.mean_epipolar_error}"
    )


# --------------------------------------------------------------------------
# Test 3: epipolar direction sanity (convention pinned down)
# --------------------------------------------------------------------------
def test_epipolar_direction_convention():
    pts_a, pts_b = _multidepth_points()
    F_gt = _ground_truth_F()

    err_correct = mean_symmetric_epipolar_error(F_gt, pts_a, pts_b)
    err_transposed = mean_symmetric_epipolar_error(F_gt.T, pts_a, pts_b)

    # Correct convention (pts_b^T F pts_a = 0) is near-zero on exact points;
    # the transposed F violates it badly. If these were swapped, the direction
    # would be silently inverted.
    assert err_correct <= 1e-2, f"correct convention not near zero: {err_correct}"
    assert err_transposed > 1.0, (
        f"transposed F should give large error, got {err_transposed}"
    )
    assert err_transposed > err_correct


# --------------------------------------------------------------------------
# Test 4: insufficient matches
# --------------------------------------------------------------------------
def test_too_few_point_correspondences_raises():
    pts_a, pts_b = _multidepth_points(n=5)  # below the 8-point floor
    with pytest.raises(InsufficientGeometryError):
        estimate_fundamental_from_points(pts_a, pts_b)


def test_textureless_image_pair_raises():
    rng = np.random.default_rng(7)
    # Near-uniform images: almost no SIFT features -> too few matches.
    img_a = np.full((IMG_H, IMG_W), 120, dtype=np.uint8)
    img_b = np.full((IMG_H, IMG_W), 120, dtype=np.uint8)
    img_a[0, 0] = 121
    img_b[0, 0] = 121
    with pytest.raises(InsufficientGeometryError):
        estimate_two_view_geometry(img_a, img_b)


# --------------------------------------------------------------------------
# Test 5: degenerate / planar scene
# --------------------------------------------------------------------------
def test_planar_scene_is_refused():
    pts_a, pts_b = _planar_points()
    with pytest.raises(InsufficientGeometryError):
        estimate_fundamental_from_points(pts_a, pts_b)


def test_pure_noise_second_image_is_refused():
    img_a = _texture(11)
    rng = np.random.default_rng(424242)
    img_b = rng.integers(0, 256, size=(IMG_H, IMG_W), dtype=np.uint8)
    with pytest.raises(InsufficientGeometryError):
        estimate_two_view_geometry(img_a, img_b)


# --------------------------------------------------------------------------
# Test 6: input validation
# --------------------------------------------------------------------------
def test_rejects_non_2d_array():
    img_a = _texture(1)
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_two_view_geometry(img_a, bad)


def test_rejects_empty_array():
    img_a = _texture(1)
    empty = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_two_view_geometry(img_a, empty)


def test_rejects_mismatched_dtypes():
    img_a = _texture(1)
    img_b = img_a.astype(np.uint16)
    with pytest.raises(ValueError):
        estimate_two_view_geometry(img_a, img_b)


# --------------------------------------------------------------------------
# Test 7: result contract
# --------------------------------------------------------------------------
def test_two_view_geometry_contract():
    img_a, img_b = _multidepth_image_pair()
    geom = estimate_two_view_geometry(img_a, img_b)

    assert isinstance(geom, TwoViewGeometry)
    assert isinstance(geom.fundamental_matrix, np.ndarray)
    assert geom.fundamental_matrix.shape == (3, 3)
    assert isinstance(geom.inlier_count, int)
    assert isinstance(geom.mean_epipolar_error, float)
    assert math.isfinite(geom.mean_epipolar_error)
    assert isinstance(geom.num_matches_used, int)
    assert geom.num_matches_used >= geom.inlier_count
