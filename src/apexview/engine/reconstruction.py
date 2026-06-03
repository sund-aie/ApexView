"""Relative two-view pose recovery and triangulation for ApexView.

Layer A2 of angulation handling. Given an angulated radiograph pair, recover
the relative camera pose (R, t) and a RELATIVE-scale 3D point cloud of the
matched inlier correspondences. Builds directly on
:mod:`apexview.engine.stereo_geometry` (A1): F estimation and inlier
correspondences come from :func:`estimate_two_view_geometry`; this module
adds the essential-matrix decomposition, pose recovery, and triangulation
on top.

The reconstruction is RELATIVE: two uncalibrated views recover geometry only
up to an unknown global scale. Every public name, parameter, and metric in
this module says "relative" because that is the truth. Absolute / metric /
millimetre scale is a deliberate later task that needs a scale reference
(sensor specs or calibration object) — it is NOT computed here, and the
output must not be mistaken for one. Likewise, this module performs NO
image rectification, NO corrected-image output, NO measurement.

Calibration: triangulation needs intrinsics. Real intraoral DICOMs frequently
do not carry true intrinsics, so we handle this honestly via
:func:`assumed_intrinsics`, which builds a nominal K from image dimensions
and a documented heuristic focal length. The caller threads the
"intrinsics were assumed" flag into the result so any downstream consumer
knows the geometry is only as good as that assumption.

Single source of truth: ``rotation``, ``translation``, ``points_3d``, and
``reprojection_error_px`` are computed once, here, and stored on
:class:`RelativeReconstruction`. Any future consumer reads these fields and
never recomputes them. CLAHE preprocessing continues to come ONLY from the
A1 path (:func:`preprocess_for_matching` via
:func:`estimate_two_view_geometry`); this module does not preprocess again.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.stereo_geometry import (
    InsufficientGeometryError,
    estimate_two_view_geometry,
)

# Minimum points surviving recoverPose's cheirality check for the
# reconstruction to be considered trustworthy. The 8-point algorithm's
# mathematical floor; below this, the geometry is not reliably determined.
_MIN_CHEIRALITY_INLIERS = 8

# Implausibly-large reprojection error sanity guard, in pixels. On clean
# synthetic data the round-trip is sub-pixel; even noisy real radiographs
# with assumed intrinsics stay well under this. A value above it indicates a
# broken reconstruction (e.g. wrong cheirality branch, numerical collapse).
_MAX_REPROJ_ERROR_PX = 50.0


class ReconstructionError(Exception):
    """Raised when relative pose or triangulation cannot be reliably
    recovered: too few cheirality-passing inliers, or an implausibly large
    reprojection error on the triangulated points.

    Note: upstream geometry problems (too few SIFT matches, degenerate/planar
    scene, RANSAC failure) propagate as :class:`InsufficientGeometryError`
    from the A1 layer, unwrapped. Callers should be prepared to catch both.
    """


@dataclass
class RelativeReconstruction:
    rotation: np.ndarray              # 3x3, second camera relative to first
    translation: np.ndarray           # length-3, UNIT-NORM direction; scale unknown
    points_3d: np.ndarray             # (N, 3) in the first camera's frame, RELATIVE scale
    num_points: int                   # number of triangulated points after cheirality
    reprojection_error_px: float      # mean over both views and all points
    intrinsics_were_assumed: bool     # True if K came from assumed_intrinsics


def assumed_intrinsics(
    width: int, height: int, focal_px: float | None = None
) -> tuple[np.ndarray, bool]:
    """Build a NOMINAL intrinsics matrix K from image dimensions.

    This is an ASSUMPTION, not a calibration. If ``focal_px`` is not provided
    it defaults to ``max(width, height)`` — the standard "approximate
    intrinsics from image size" heuristic. The principal point is placed at
    the image center.

    THE RESULTING RECONSTRUCTION IS CORRECT ONLY UP TO SIMILARITY/SCALE AND
    IS ONLY AS GOOD AS THE ASSUMED FOCAL LENGTH. True intrinsics or a
    physical scale reference are required before any real-world measurement
    can be derived from a reconstruction built with this K. The second
    return value is a boolean ``True`` flag that the caller is expected to
    thread into :func:`estimate_pose_and_triangulate` (and through into the
    :class:`RelativeReconstruction`) so downstream consumers know the
    intrinsics were assumed.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"assumed_intrinsics needs positive dimensions, got {width}x{height}"
        )
    f = float(max(width, height)) if focal_px is None else float(focal_px)
    if f <= 0:
        raise ValueError(f"focal_px must be positive, got {focal_px}")
    K = np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, True


def _validate_K(K: np.ndarray) -> np.ndarray:
    if not isinstance(K, np.ndarray):
        raise ValueError("K must be a numpy ndarray")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be a 3x3 matrix, got shape {K.shape}")
    return K


def _validate_images(image_a: np.ndarray, image_b: np.ndarray) -> None:
    """Same checks the rest of the engine uses for image pairs."""
    for name, img in (("image_a", image_a), ("image_b", image_b)):
        if not isinstance(img, np.ndarray):
            raise ValueError(f"{name} must be a numpy ndarray")
        if img.ndim != 2:
            raise ValueError(f"{name} must be 2D grayscale, got shape {img.shape}")
        if img.size == 0:
            raise ValueError(f"{name} must not be empty")
    if image_a.dtype != image_b.dtype:
        raise ValueError(
            f"image_a and image_b must share dtype, got "
            f"{image_a.dtype} vs {image_b.dtype}"
        )


def _mean_reprojection_error(
    points_3d: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> float:
    """Mean Euclidean pixel distance from observed 2D points to the
    reprojections of their triangulated 3D positions, averaged over both
    views and all points."""
    n = points_3d.shape[0]
    if n == 0:
        return float("nan")
    hom = np.hstack([points_3d, np.ones((n, 1))])  # (N, 4)
    proj_a = (P1 @ hom.T).T  # (N, 3)
    proj_b = (P2 @ hom.T).T
    proj_a_2d = proj_a[:, :2] / proj_a[:, 2:3]
    proj_b_2d = proj_b[:, :2] / proj_b[:, 2:3]
    err_a = np.linalg.norm(proj_a_2d - pts_a, axis=1)
    err_b = np.linalg.norm(proj_b_2d - pts_b, axis=1)
    return float(0.5 * (err_a.mean() + err_b.mean()))


def reconstruct_from_points(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    fundamental_matrix: np.ndarray,
    K: np.ndarray,
    *,
    intrinsics_were_assumed: bool = False,
) -> RelativeReconstruction:
    """Recover relative pose + triangulate from explicit correspondences.

    Pure math: no images, no SIFT. Used by :func:`estimate_pose_and_triangulate`
    after A1 has produced the correspondences and F, and exposed as a public
    helper so the math layer can be tested without SIFT noise.

    ``pts_a`` and ``pts_b`` are matched 2D image coordinates in image_a /
    image_b respectively (matching the convention ``pts_b^T F pts_a = 0``).
    """
    K = _validate_K(K)
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if pa.shape != pb.shape:
        raise ValueError(
            f"pts_a and pts_b must have the same shape, got {pa.shape} vs {pb.shape}"
        )
    if pa.shape[0] < _MIN_CHEIRALITY_INLIERS:
        raise ReconstructionError(
            f"Only {pa.shape[0]} correspondences; need at least "
            f"{_MIN_CHEIRALITY_INLIERS} for trustworthy two-view pose recovery."
        )

    F = np.asarray(fundamental_matrix, dtype=np.float64)
    if F.shape != (3, 3):
        raise ReconstructionError(
            f"Fundamental matrix must be 3x3, got shape {F.shape}."
        )

    # E = K^T F K relates pixel-normalized correspondences.
    E = K.T @ F @ K

    # recoverPose decomposes E and selects the (R, t) branch consistent with
    # cheirality (points in front of both cameras). Returned t is unit-norm
    # because scale is fundamentally unknowable from two uncalibrated views.
    pa_cv = pa.reshape(-1, 1, 2)
    pb_cv = pb.reshape(-1, 1, 2)
    retval, R, t, cheir_mask = cv2.recoverPose(E, pa_cv, pb_cv, K)
    if R is None or t is None or cheir_mask is None:
        raise ReconstructionError(
            "cv2.recoverPose returned no usable pose (degenerate essential matrix)."
        )
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    # recoverPose's mask marks points that passed cheirality; only those are
    # safe to triangulate. Below the floor we refuse the reconstruction.
    mask_flat = cheir_mask.ravel().astype(bool)
    cheir_count = int(mask_flat.sum())
    if cheir_count < _MIN_CHEIRALITY_INLIERS:
        raise ReconstructionError(
            f"Only {cheir_count} correspondences survived cheirality; need at "
            f"least {_MIN_CHEIRALITY_INLIERS} for a trustworthy reconstruction."
        )
    pa_ok = pa[mask_flat]
    pb_ok = pb[mask_flat]

    # Triangulate via the standard two-view linear method.
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pa_ok.T, pb_ok.T)  # (4, N)
    w = pts4d[3]
    if np.any(np.abs(w) < 1e-12):
        raise ReconstructionError(
            "Triangulation produced points at infinity; geometry is degenerate."
        )
    pts3d = (pts4d[:3] / w).T  # (N, 3) in camera_a's frame

    reproj_err = _mean_reprojection_error(pts3d, pa_ok, pb_ok, P1, P2)
    if not np.isfinite(reproj_err) or reproj_err > _MAX_REPROJ_ERROR_PX:
        raise ReconstructionError(
            f"Implausible reprojection error {reproj_err:.3f} px (limit "
            f"{_MAX_REPROJ_ERROR_PX}). Reconstruction is not trustworthy."
        )

    return RelativeReconstruction(
        rotation=R,
        translation=t / np.linalg.norm(t) if np.linalg.norm(t) > 0 else t,
        points_3d=pts3d,
        num_points=int(pts3d.shape[0]),
        reprojection_error_px=reproj_err,
        intrinsics_were_assumed=bool(intrinsics_were_assumed),
    )


def estimate_pose_and_triangulate(
    image_a: np.ndarray,
    image_b: np.ndarray,
    K: np.ndarray,
    *,
    apply_clahe: bool = True,
    intrinsics_were_assumed: bool = False,
) -> RelativeReconstruction:
    """Recover relative pose + triangulate inlier matches from a radiograph pair.

    Uses :func:`estimate_two_view_geometry` (A1) for SIFT matching, RANSAC
    fundamental-matrix estimation, the planar-degeneracy guard, and CLAHE
    preprocessing (via the ``apply_clahe`` flag — same default-on contract
    as the rest of the engine). The returned reconstruction is RELATIVE in
    scale; if ``K`` came from :func:`assumed_intrinsics`, set
    ``intrinsics_were_assumed=True`` so the output surfaces that honestly.

    Raises:
        ValueError: invalid images or K.
        InsufficientGeometryError: too few matches, planar/degenerate scene,
            RANSAC failure (propagated from A1).
        ReconstructionError: pose/triangulation failed (too few cheirality
            inliers, implausible reprojection error, etc.).
    """
    _validate_images(image_a, image_b)
    K = _validate_K(K)

    geom = estimate_two_view_geometry(image_a, image_b, apply_clahe=apply_clahe)
    return reconstruct_from_points(
        geom.inlier_points_a,
        geom.inlier_points_b,
        geom.fundamental_matrix,
        K,
        intrinsics_were_assumed=intrinsics_were_assumed,
    )
