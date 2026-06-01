"""Two-view stereo geometry for ApexView: fundamental matrix estimation.

This is the first layer of angulation handling. It recovers the relative
two-view epipolar geometry (the fundamental matrix F) of an angulated
radiograph pair. It is foundation math only — NO essential matrix, pose,
triangulation, 3D points, rectification, correction, or measurement. Those are
later tasks (A2/A3).

Matching convention (mirrors :mod:`apexview.engine.extension_stitch` exactly,
so the whole codebase shares one convention): SIFT via ``cv2.SIFT_create``,
``cv2.BFMatcher(cv2.NORM_L2)``, ``knnMatch(des_b, des_a, k=2)``, Lowe ratio
0.75. Points in ``image_a`` are the left / first-camera points (``pts_a``);
points in ``image_b`` are the right / second-camera points (``pts_b``).

Epipolar direction (pinned down by tests, because an inverted convention is a
classic silent bug): F satisfies

    pts_b^T  F  pts_a = 0

i.e. F maps a point in image_a to its epipolar line in image_b. This is the
convention OpenCV's ``findFundamentalMat(points1=pts_a, points2=pts_b)``
returns.

Single source of truth: ``mean_epipolar_error`` and ``inlier_count`` are
computed once, here, and stored on :class:`TwoViewGeometry`. Any future
consumer (A2/A3, a UI) reads those fields and must NOT recompute them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.preprocessing import preprocess_for_matching

# Mirror extension_stitch's matching convention exactly.
_LOWE_RATIO = 0.75

# The 8-point algorithm's hard mathematical floor. Quality needs many more,
# well-distributed correspondences; 8 is enforced as the absolute minimum.
_MIN_CORRESPONDENCES = 8

# RANSAC parameters for findFundamentalMat. Threshold is the max distance (px)
# from a point to its epipolar line for the pair to count as an inlier.
_RANSAC_THRESHOLD_PX = 2.0
_RANSAC_CONFIDENCE = 0.99

# Planar-degeneracy guard. A fundamental matrix is ill-defined when the scene
# is planar (or the motion is a pure rotation): then a single homography
# explains the correspondences just as well, and F is one arbitrary member of
# an ambiguous family. cv2 returns such an F confidently (full inlier support,
# tiny self-consistent error), so shape/inlier guards do NOT catch it. We
# detect it the standard way (as in ORB-SLAM / COLMAP): fit a homography to the
# same correspondences and compare its inlier support to F's. If a homography
# explains at least this fraction of F's inliers, the geometry is degenerate
# and we refuse. Measured separation on synthetic data is wide (~0.23 for a
# genuine multi-depth scene vs ~1.0 for a planar one), so 0.90 is safely
# inside the gap, not a tuned knob.
_DEGENERACY_HOMOGRAPHY_RATIO = 0.90


class InsufficientGeometryError(Exception):
    """Raised when two-view geometry cannot be reliably estimated: too few
    correspondences, RANSAC failure, a degenerate/non-3x3 result, or an
    inlier set below the 8-point minimum."""


@dataclass
class TwoViewGeometry:
    fundamental_matrix: np.ndarray
    inlier_count: int
    mean_epipolar_error: float
    num_matches_used: int


def _validate(image_a: np.ndarray, image_b: np.ndarray) -> None:
    """Input validation, mirroring extension_stitch._validate exactly."""
    for name, img in (("image_a", image_a), ("image_b", image_b)):
        if not isinstance(img, np.ndarray):
            raise ValueError(f"{name} must be a numpy ndarray")
        if img.ndim != 2:
            raise ValueError(f"{name} must be 2D grayscale, got shape {img.shape}")
        if img.size == 0:
            raise ValueError(f"{name} must not be empty")
    if image_a.dtype != image_b.dtype:
        raise ValueError(
            f"image_a and image_b must share dtype, got {image_a.dtype} vs {image_b.dtype}"
        )


def _match_points(
    image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + Lowe-ratio match, mirroring extension_stitch exactly.

    knnMatch(des_b, des_a) means each match's queryIdx indexes image_b's
    keypoints and trainIdx indexes image_a's. Returns ``(pts_a, pts_b)`` as
    ``(N, 2)`` float32 arrays of corresponding points.
    """
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_b, des_a, k=2)
    good = [
        m
        for pair in knn
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < _LOWE_RATIO * n.distance
    ]

    pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts_a, pts_b


def mean_symmetric_epipolar_error(
    fundamental_matrix: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray
) -> float:
    """Mean symmetric epipolar distance (px) over the given correspondences.

    For each pair, with F satisfying ``pts_b^T F pts_a = 0``:
      * line in image_b: ``l_b = F @ [pts_a; 1]``; distance of pts_b to l_b
      * line in image_a: ``l_a = F^T @ [pts_b; 1]``; distance of pts_a to l_a
    Each distance is the standard point-to-line distance, normalized by
    ``sqrt(a^2 + b^2)`` of the line. The per-pair value is the average of the
    two one-way distances; the result is the mean over all pairs.
    """
    F = np.asarray(fundamental_matrix, dtype=np.float64)
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pa.shape[0], 1), dtype=np.float64)
    ha = np.hstack([pa, ones])  # N x 3 homogeneous image_a points
    hb = np.hstack([pb, ones])  # N x 3 homogeneous image_b points

    lines_b = ha @ F.T  # each row = F @ x_a  -> epipolar line in image_b
    lines_a = hb @ F  # each row = F^T @ x_b -> epipolar line in image_a

    eps = 1e-12
    d_b = np.abs(np.sum(lines_b * hb, axis=1)) / np.maximum(
        np.sqrt(lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2), eps
    )
    d_a = np.abs(np.sum(lines_a * ha, axis=1)) / np.maximum(
        np.sqrt(lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2), eps
    )
    return float(np.mean(0.5 * (d_b + d_a)))


def estimate_fundamental_from_points(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F from explicit correspondences via RANSAC 8-point.

    ``pts_a`` are image_a (first-camera) points, ``pts_b`` are image_b
    (second-camera) points; returned F satisfies ``pts_b^T F pts_a = 0``.
    Returns ``(F, inlier_mask)`` where inlier_mask is a boolean array over the
    input correspondences. Raises InsufficientGeometryError if fewer than 8
    correspondences, RANSAC fails, the result is not a single 3x3, or fewer
    than 8 inliers support the model.
    """
    pa = np.asarray(pts_a, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float32).reshape(-1, 2)
    if pa.shape != pb.shape:
        raise ValueError(
            f"pts_a and pts_b must have the same shape, got {pa.shape} vs {pb.shape}"
        )

    n = pa.shape[0]
    if n < _MIN_CORRESPONDENCES:
        raise InsufficientGeometryError(
            f"Only {n} correspondences; the 8-point algorithm needs at least "
            f"{_MIN_CORRESPONDENCES}."
        )

    F, mask = cv2.findFundamentalMat(
        pa, pb, cv2.FM_RANSAC, _RANSAC_THRESHOLD_PX, _RANSAC_CONFIDENCE
    )

    if F is None or mask is None or F.size == 0:
        raise InsufficientGeometryError(
            "findFundamentalMat returned no usable model (degenerate geometry)."
        )
    F = np.asarray(F, dtype=np.float64)
    # cv2 can return a stacked 9x3 (multiple candidate solutions) for
    # degenerate inputs; only a single 3x3 is acceptable here.
    if F.shape != (3, 3):
        raise InsufficientGeometryError(
            f"Expected a single 3x3 fundamental matrix, got shape {F.shape} "
            f"(degenerate geometry)."
        )

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _MIN_CORRESPONDENCES:
        raise InsufficientGeometryError(
            f"RANSAC supported F with only {inlier_count} inliers; need at "
            f"least {_MIN_CORRESPONDENCES} for a trustworthy model."
        )

    # Planar-degeneracy guard (see _DEGENERACY_HOMOGRAPHY_RATIO).
    H, h_mask = cv2.findHomography(pa, pb, cv2.RANSAC, _RANSAC_THRESHOLD_PX)
    h_inliers = int(h_mask.sum()) if h_mask is not None else 0
    if h_inliers >= _DEGENERACY_HOMOGRAPHY_RATIO * inlier_count:
        raise InsufficientGeometryError(
            f"Scene appears planar/degenerate: a homography explains "
            f"{h_inliers} correspondences vs F's {inlier_count} inliers "
            f"(ratio {h_inliers / inlier_count:.2f} >= "
            f"{_DEGENERACY_HOMOGRAPHY_RATIO}). The fundamental matrix is not "
            f"reliably determined."
        )
    return F, inlier_mask


def estimate_two_view_geometry(
    image_a: np.ndarray, image_b: np.ndarray, apply_clahe: bool = True
) -> TwoViewGeometry:
    """Estimate the fundamental matrix relating two radiographs.

    With ``apply_clahe=True`` (the default), each input image is passed
    through the shared :func:`preprocess_for_matching` before SIFT. CLAHE
    only rescales intensities, so the matched keypoint coordinates and the
    resulting fundamental matrix remain in the caller's image coordinate
    frame; nothing about the geometry math changes. The caller's input
    arrays are never mutated. Pass ``apply_clahe=False`` to isolate
    raw-input behavior.

    Raises:
        ValueError: invalid inputs (non-2D, empty, or mismatched dtypes).
        InsufficientGeometryError: too few matches, RANSAC failure, a
            degenerate/non-3x3 result, or an inlier set below the 8-point
            minimum.
    """
    _validate(image_a, image_b)
    match_a = preprocess_for_matching(image_a, apply_clahe=apply_clahe)
    match_b = preprocess_for_matching(image_b, apply_clahe=apply_clahe)
    pts_a, pts_b = _match_points(match_a, match_b)

    if pts_a.shape[0] < _MIN_CORRESPONDENCES:
        raise InsufficientGeometryError(
            f"Only {pts_a.shape[0]} good matches after Lowe ratio test; the "
            f"8-point algorithm needs at least {_MIN_CORRESPONDENCES}."
        )

    F, inlier_mask = estimate_fundamental_from_points(pts_a, pts_b)
    inlier_count = int(inlier_mask.sum())
    mean_error = mean_symmetric_epipolar_error(
        F, pts_a[inlier_mask], pts_b[inlier_mask]
    )

    return TwoViewGeometry(
        fundamental_matrix=F,
        inlier_count=inlier_count,
        mean_epipolar_error=mean_error,
        num_matches_used=int(pts_a.shape[0]),
    )
