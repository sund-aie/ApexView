"""Extension stitcher for ApexView.

Two intra-oral radiographs taken at the SAME beam angle but with the sensor
translated sideways form a planar "extension" pair: a single planar homography
relates them. This module recovers that homography from SIFT matches refined
by RANSAC, warps the source onto the destination's frame, and reports the mean
reprojection error of the RANSAC inliers.

The mean reprojection error is the quantitative signal that the pair really is
a planar extension. A future angulation-correction engine will use a high
value here to decide that the input pair is NOT a flat extension and route the
images through epipolar/triangulation logic instead. Because that decision
depends on this number being trustworthy, the value MUST come from this engine
unmodified.

Single source of truth: ``mean_reprojection_error`` and ``inlier_count`` on
:class:`StitchResult` are computed here, once, by the engine. Any future UI is
a thin client and must read those fields verbatim — it must never recompute
them from the returned image or homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.preprocessing import preprocess_for_matching

_LOWE_RATIO = 0.75
_MIN_MATCHES = 4
_RANSAC_REPROJ_THRESHOLD = 3.0


class InsufficientOverlapError(Exception):
    """Raised when the two images do not share enough features to align.

    Triggered if Lowe-filtered matches fall below the 4-point minimum required
    by a planar homography, or if RANSAC fails to find a model with at least
    4 inliers.
    """


@dataclass
class StitchResult:
    stitched_image: np.ndarray
    homography: np.ndarray
    inlier_count: int
    mean_reprojection_error: float


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


def stitch_extension(
    image_a: np.ndarray, image_b: np.ndarray, apply_clahe: bool = True
) -> StitchResult:
    """Stitch two extension-pair grayscale radiographs into one wider image.

    image_a is treated as the reference frame. image_b is warped into image_a's
    coordinate system via a homography recovered from SIFT + Lowe + RANSAC.

    With ``apply_clahe=True`` (the default), each input image is passed
    through the shared :func:`preprocess_for_matching` before SIFT, which
    materially improves inlier spatial coverage on real radiographs. The
    preprocessing is applied to the SIFT-input copies only; the stitched
    output canvas is built from the caller's ORIGINAL pixels, and the
    caller's input arrays are never mutated. Pass ``apply_clahe=False`` to
    feed SIFT the raw images (useful for tests that isolate raw-input math).

    Raises:
        ValueError: invalid inputs (non-2D, empty, or mismatched dtypes).
        InsufficientOverlapError: not enough matchable features, or RANSAC
            could not find a model with at least 4 inliers.
    """
    _validate(image_a, image_b)

    match_a = preprocess_for_matching(image_a, apply_clahe=apply_clahe)
    match_b = preprocess_for_matching(image_b, apply_clahe=apply_clahe)

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(match_a, None)
    kp_b, des_b = sift.detectAndCompute(match_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        raise InsufficientOverlapError(
            "Not enough SIFT features detected to attempt alignment."
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_b, des_a, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < _LOWE_RATIO * n.distance]

    if len(good) < _MIN_MATCHES:
        raise InsufficientOverlapError(
            f"Only {len(good)} good matches after Lowe ratio test; need at least {_MIN_MATCHES}."
        )

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD
    )
    if homography is None or mask is None:
        raise InsufficientOverlapError("RANSAC failed to estimate a homography.")

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _MIN_MATCHES:
        raise InsufficientOverlapError(
            f"RANSAC produced only {inlier_count} inliers; need at least {_MIN_MATCHES}."
        )

    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    projected = cv2.perspectiveTransform(src_inliers, homography)
    errors = np.linalg.norm(projected - dst_inliers, axis=2).ravel()
    mean_reprojection_error = float(errors.mean())

    stitched = _compose(image_a, image_b, homography)

    return StitchResult(
        stitched_image=stitched,
        homography=homography,
        inlier_count=inlier_count,
        mean_reprojection_error=mean_reprojection_error,
    )


def _compose(
    image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape

    corners_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
    corners_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
    warped_b_corners = cv2.perspectiveTransform(corners_b, homography)

    all_corners = np.concatenate([corners_a, warped_b_corners], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translation = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    canvas_w = int(x_max - x_min)
    canvas_h = int(y_max - y_min)

    warped_b = cv2.warpPerspective(
        image_b, translation @ homography, (canvas_w, canvas_h)
    )
    canvas = warped_b.copy()
    canvas[
        -y_min : -y_min + h_a,
        -x_min : -x_min + w_a,
    ] = image_a

    return canvas
