"""Rough overlap alignment of two angled radiographs for a pivot viewer.

Intra-oral pairs that the classifier flags as ANGULATION are two real
radiographs of the same teeth taken at different beam angles. No single
planar warp can stitch them — that's the parallax — and the extension
stitcher correctly refuses. This module does NOT try to stitch them. It
fits one homography from the shared anatomy and uses it to project
``image_b`` into ``image_a``'s pixel grid so a viewer can flip between the
two real angles with rough overlap. Residual misalignment after the warp
is the angle difference itself; it is NOT a defect of the alignment and
must not be claimed as one.

This is NOT a stitch, NOT an angulation correction, NOT a 3D
reconstruction. It is a comparison-view registration only.

Reuses the existing matching recipe verbatim — SIFT via
``cv2.SIFT_create``, ``BFMatcher`` with ``NORM_L2``,
``knnMatch(des_b, des_a, k=2)`` with Lowe ratio 0.75,
``cv2.findHomography`` with RANSAC and a 3.0 px threshold — and runs both
inputs through :func:`apexview.engine.preprocessing.preprocess_for_matching`
with CLAHE on, the same conditioning the rest of the engine uses. Nothing
is factored out of :mod:`apexview.engine.extension_stitch`; that file is
left untouched.

Single source of truth: ``warped_b_to_a``, ``inlier_count``, and
``mean_alignment_error_px`` on :class:`PivotAlignment` are computed here
once and returned to callers verbatim. A GUI must not recompute them from
the image or the homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.preprocessing import preprocess_for_matching

_LOWE_RATIO = 0.75
_MIN_MATCHES = 4
_RANSAC_REPROJ_THRESHOLD = 3.0


class PivotAlignmentError(Exception):
    """Raised when even a rough overlap alignment cannot be fit.

    Triggered if SIFT cannot find enough features, Lowe-filtered matches
    fall below the 4-point homography minimum, or RANSAC fails to produce
    a model with at least 4 inliers. The pivot viewer should fall back to
    a plain side-by-side display on this error and surface the message.
    """


@dataclass(frozen=True)
class PivotAlignment:
    warped_b_to_a: np.ndarray
    image_a: np.ndarray
    inlier_count: int
    mean_alignment_error_px: float
    homography: np.ndarray


def _validate(image_a: np.ndarray, image_b: np.ndarray) -> None:
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


def align_for_pivot(image_a: np.ndarray, image_b: np.ndarray) -> PivotAlignment:
    """Roughly register ``image_b`` onto ``image_a`` for comparison viewing.

    Both inputs are uint8 2D grayscale arrays. They are passed through
    :func:`preprocess_for_matching` with CLAHE on, then SIFT + Lowe +
    RANSAC fits a homography mapping ``image_b`` -> ``image_a``. The warp
    is applied with ``cv2.warpPerspective`` at ``image_a``'s ``(H, W)`` so
    the result drops into the reference frame without changing canvas
    size. The returned ``mean_alignment_error_px`` is the mean reprojection
    error of the RANSAC inliers and is the honesty signal a UI should show
    alongside the viewer.

    This is a comparison-view registration of two different-angle real
    radiographs. Residual misalignment after the warp IS the parallax of
    the angle difference. It is NOT a stitch, NOT an angulation correction,
    NOT a 3D reconstruction.

    Raises:
        ValueError: invalid inputs (non-2D, empty, or mismatched dtypes).
        PivotAlignmentError: not enough matchable features, or RANSAC
            could not find a model with at least 4 inliers.
    """
    _validate(image_a, image_b)

    match_a = preprocess_for_matching(image_a, apply_clahe=True)
    match_b = preprocess_for_matching(image_b, apply_clahe=True)

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(match_a, None)
    kp_b, des_b = sift.detectAndCompute(match_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        raise PivotAlignmentError(
            "Not enough SIFT features detected to attempt rough alignment."
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_b, des_a, k=2)
    good = [
        m for pair in knn
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < _LOWE_RATIO * n.distance
    ]

    if len(good) < _MIN_MATCHES:
        raise PivotAlignmentError(
            f"Only {len(good)} good matches after Lowe ratio test; "
            f"need at least {_MIN_MATCHES} for a rough alignment."
        )

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD
    )
    if homography is None or mask is None:
        raise PivotAlignmentError(
            "RANSAC failed to estimate an alignment homography."
        )

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _MIN_MATCHES:
        raise PivotAlignmentError(
            f"RANSAC produced only {inlier_count} inliers; "
            f"need at least {_MIN_MATCHES} for a rough alignment."
        )

    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    projected = cv2.perspectiveTransform(src_inliers, homography)
    errors = np.linalg.norm(projected - dst_inliers, axis=2).ravel()
    mean_alignment_error_px = float(errors.mean())

    h_a, w_a = image_a.shape
    warped_b_to_a = cv2.warpPerspective(image_b, homography, (w_a, h_a))

    return PivotAlignment(
        warped_b_to_a=warped_b_to_a,
        image_a=image_a,
        inlier_count=inlier_count,
        mean_alignment_error_px=mean_alignment_error_px,
        homography=homography,
    )
