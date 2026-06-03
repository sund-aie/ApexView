"""Tests for relative two-view pose recovery and triangulation.

Fully synthetic and deterministic, mirroring the structure of A1:

  * EXACT-MATH tests project known multi-depth 3D points through two cameras
    with known intrinsics K and known relative pose (R_true, t_true), giving
    exact correspondences and an analytically-constructed ground-truth F.
    These pin pose recovery and triangulation to tight tolerance.
  * END-TO-END test renders the multi-depth procedural image pair from A1
    and runs the full image-level pipeline (SIFT through reconstruction),
    tolerating SIFT noise.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from apexview.engine.reconstruction import (
    RelativeReconstruction,
    ReconstructionError,
    assumed_intrinsics,
    estimate_pose_and_triangulate,
    reconstruct_from_points,
)
from apexview.engine.stereo_geometry import (
    InsufficientGeometryError,
)

IMG_H = 480
IMG_W = 640

K_A = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
K_B = np.array([[820.0, 0.0, 310.0], [0.0, 820.0, 250.0], [0.0, 0.0, 1.0]])
# Single calibrated K used for the exact-math tests (same camera both views).
K_SHARED = K_A

_THETA = math.radians(8.0)
R_TRUE = np.array(
    [
        [math.cos(_THETA), 0.0, math.sin(_THETA)],
        [0.0, 1.0, 0.0],
        [-math.sin(_THETA), 0.0, math.cos(_THETA)],
    ]
)
T_TRUE = np.array([0.6, 0.05, 0.1])


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]
    )


def _ground_truth_F(K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    F = np.linalg.inv(K2).T @ _skew(T_TRUE) @ R_TRUE @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = (R @ X.T).T + t
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


def _multidepth_points(seed: int = 0, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact correspondences and the underlying 3D points (for depth-order checks)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, n),
            rng.uniform(-1.5, 1.5, n),
            rng.uniform(4.0, 12.0, n),
        ]
    )
    pa = _project(K_SHARED, np.eye(3), np.zeros(3), X)
    pb = _project(K_SHARED, R_TRUE, T_TRUE, X)
    return pa, pb, X


def _planar_points(seed: int = 1, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = 3.0 * x + 2.0 * y + 8.0
    X = np.column_stack([x, y, z])
    pa = _project(K_SHARED, np.eye(3), np.zeros(3), X)
    pb = _project(K_SHARED, R_TRUE, T_TRUE, X)
    return pa, pb


# --------------------------------------------------------------------------
# Multi-depth procedural image pair (mirrors the A1 fixture).
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
    return K_B @ (R_TRUE - np.outer(T_TRUE, [0.0, 0.0, 1.0]) / depth_z) @ np.linalg.inv(K_A)


def _multidepth_image_pair(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
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
# Test 1 — known-pose recovery on exact correspondences
# --------------------------------------------------------------------------
def test_known_pose_recovery_from_exact_points():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)

    recon = reconstruct_from_points(pa, pb, F, K_SHARED)

    # Rotation: extract the angle of the residual R_true^T R_rec.
    R_rel = R_TRUE.T @ recon.rotation
    cos_angle = max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))
    angle_err_deg = math.degrees(math.acos(cos_angle))
    assert angle_err_deg < 1.0, f"rotation angle error too high: {angle_err_deg} deg"

    # Translation: direction (sign-insensitive) must align with t_true.
    t_n_true = T_TRUE / np.linalg.norm(T_TRUE)
    t_n_rec = recon.translation / np.linalg.norm(recon.translation)
    cos_t = float(abs(np.dot(t_n_true, t_n_rec)))
    assert cos_t > 0.999, f"translation direction cosine too low: {cos_t}"


# --------------------------------------------------------------------------
# Test 2 — known-structure (self-consistency + depth ordering)
# --------------------------------------------------------------------------
def test_known_structure_self_consistent_and_depth_ordered():
    pa, pb, X_true = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)

    recon = reconstruct_from_points(pa, pb, F, K_SHARED)

    # Self-consistency: triangulated points reproject to within 1e-2 px of the
    # observed 2D correspondences. On exact input this is essentially round-off.
    assert recon.reprojection_error_px < 1e-2, (
        f"reprojection error too high on exact input: {recon.reprojection_error_px}"
    )

    # Depth ordering: rank correlation of true Z vs recovered Z is +1. A
    # mirrored / garbage reconstruction would scramble this.
    order_true = np.argsort(X_true[:, 2])
    order_rec = np.argsort(recon.points_3d[:, 2])
    assert np.array_equal(order_true, order_rec), (
        "recovered depth ordering does not match the known ordering"
    )


# --------------------------------------------------------------------------
# Test 3 — cheirality / sign sanity
# --------------------------------------------------------------------------
def test_cheirality_places_points_in_front_of_both_cameras():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    recon = reconstruct_from_points(pa, pb, F, K_SHARED)

    # Camera A frame: Z > 0 for every point.
    assert np.all(recon.points_3d[:, 2] > 0), (
        "some points have Z <= 0 in camera A; cheirality failed"
    )
    # Camera B frame: rotate-translate then check Z > 0.
    Xc_b = (recon.rotation @ recon.points_3d.T).T + recon.translation
    assert np.all(Xc_b[:, 2] > 0), (
        "some points have Z <= 0 in camera B; cheirality failed"
    )


# --------------------------------------------------------------------------
# Test 4 — assumed_intrinsics helper and flag propagation
# --------------------------------------------------------------------------
def test_assumed_intrinsics_constructs_nominal_K_and_marks_flag():
    K, flag = assumed_intrinsics(width=640, height=480)
    assert flag is True
    assert K.shape == (3, 3)
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0  # principal point at center
    assert K[0, 0] == K[1, 1] == 640.0  # nominal focal = max(w, h)
    assert K[2, 2] == 1.0
    # Off-diagonals on focal block are zero.
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0


def test_assumed_intrinsics_focal_override():
    K, flag = assumed_intrinsics(640, 480, focal_px=900.0)
    assert flag is True
    assert K[0, 0] == 900.0 and K[1, 1] == 900.0


def test_assumed_intrinsics_flag_propagates_into_reconstruction():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    K_assumed, flag = assumed_intrinsics(IMG_W, IMG_H, focal_px=K_SHARED[0, 0])
    recon = reconstruct_from_points(pa, pb, F, K_assumed, intrinsics_were_assumed=flag)
    assert recon.intrinsics_were_assumed is True


def test_flag_defaults_to_false_when_caller_does_not_set_it():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    recon = reconstruct_from_points(pa, pb, F, K_SHARED)
    assert recon.intrinsics_were_assumed is False


# --------------------------------------------------------------------------
# Test 5 — degenerate / insufficient inputs
# --------------------------------------------------------------------------
def test_too_few_correspondences_raises():
    pa, pb, _ = _multidepth_points(n=5)
    F = _ground_truth_F(K_SHARED, K_SHARED)
    with pytest.raises(ReconstructionError):
        reconstruct_from_points(pa, pb, F, K_SHARED)


def test_planar_image_pair_propagates_insufficient_geometry():
    # Synthesise an image pair from a planar scene by warping a single
    # textured plane through one homography (A->B). The planar-degeneracy
    # guard in A1 should refuse the geometry; estimate_pose_and_triangulate
    # must surface that refusal, not return a fake reconstruction.
    img_a = _texture(seed=11)
    Hab = _plane_homography(8.0)
    img_b = cv2.warpPerspective(img_a, Hab, (IMG_W, IMG_H))
    with pytest.raises(InsufficientGeometryError):
        estimate_pose_and_triangulate(img_a, img_b, K_A)


# --------------------------------------------------------------------------
# Test 6 — end-to-end through SIFT
# --------------------------------------------------------------------------
def test_end_to_end_with_assumed_intrinsics():
    img_a, img_b = _multidepth_image_pair()
    K, flag = assumed_intrinsics(IMG_W, IMG_H)
    recon = estimate_pose_and_triangulate(
        img_a, img_b, K, intrinsics_were_assumed=flag
    )

    assert isinstance(recon, RelativeReconstruction)
    assert recon.intrinsics_were_assumed is True
    assert recon.num_points >= 8
    # Realistic tolerance for SIFT localisation noise; A1's end-to-end test
    # caps mean epipolar error at 2.0, but reprojection error in triangulation
    # propagates depth uncertainty and runs higher. Empirically ~2.7 px on
    # this fixture with assumed K; 5.0 leaves margin without flattering the
    # math.
    assert recon.reprojection_error_px < 5.0, (
        f"end-to-end reprojection error too high: {recon.reprojection_error_px}"
    )


# --------------------------------------------------------------------------
# Test 7 — input validation
# --------------------------------------------------------------------------
def test_rejects_non_2d_image():
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    good = _texture(1)
    with pytest.raises(ValueError):
        estimate_pose_and_triangulate(good, bad, K_A)


def test_rejects_empty_image():
    good = _texture(1)
    empty = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_pose_and_triangulate(good, empty, K_A)


def test_rejects_mismatched_dtypes():
    img_a = _texture(1)
    img_b = img_a.astype(np.uint16)
    with pytest.raises(ValueError):
        estimate_pose_and_triangulate(img_a, img_b, K_A)


def test_rejects_non_3x3_K():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    bad_K = np.eye(4)
    with pytest.raises(ValueError):
        reconstruct_from_points(pa, pb, F, bad_K)


def test_rejects_non_array_K():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    with pytest.raises(ValueError):
        reconstruct_from_points(pa, pb, F, "not a matrix")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Test 8 — result contract
# --------------------------------------------------------------------------
def test_relative_reconstruction_contract():
    pa, pb, _ = _multidepth_points()
    F = _ground_truth_F(K_SHARED, K_SHARED)
    recon = reconstruct_from_points(pa, pb, F, K_SHARED)

    assert isinstance(recon, RelativeReconstruction)
    assert isinstance(recon.rotation, np.ndarray)
    assert recon.rotation.shape == (3, 3)
    assert isinstance(recon.translation, np.ndarray)
    assert recon.translation.shape == (3,)
    # translation should be unit-norm direction.
    assert abs(float(np.linalg.norm(recon.translation)) - 1.0) < 1e-9
    assert isinstance(recon.points_3d, np.ndarray)
    assert recon.points_3d.shape == (recon.num_points, 3)
    assert isinstance(recon.num_points, int)
    assert isinstance(recon.reprojection_error_px, float)
    assert math.isfinite(recon.reprojection_error_px)
    assert isinstance(recon.intrinsics_were_assumed, bool)


# --------------------------------------------------------------------------
# Backward-compat assertion: A1's TwoViewGeometry now exposes inlier points.
# --------------------------------------------------------------------------
def test_two_view_geometry_exposes_inlier_points_additively():
    from apexview.engine.stereo_geometry import estimate_two_view_geometry
    img_a, img_b = _multidepth_image_pair()
    geom = estimate_two_view_geometry(img_a, img_b)
    assert hasattr(geom, "inlier_points_a")
    assert hasattr(geom, "inlier_points_b")
    assert geom.inlier_points_a.shape[0] == geom.inlier_count
    assert geom.inlier_points_b.shape[0] == geom.inlier_count
    assert geom.inlier_points_a.shape[1] == 2
    assert geom.inlier_points_b.shape[1] == 2
