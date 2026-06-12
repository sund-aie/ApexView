"""Rough overlap alignment of two angled radiographs for a pivot viewer.

Intra-oral pairs that the classifier flags as ANGULATION are two real
radiographs of the same teeth taken at different beam angles. No single
planar warp can stitch them — that's the parallax — and the extension
stitcher correctly refuses. This module does NOT try to stitch them. It
fits one homography from the shared anatomy and places both radiographs
on one shared union canvas — sized to the full extent of both images,
the same corner-union construction the extension stitcher uses for its
output canvas — so a viewer can flip between the two real angles with
rough overlap and nothing cropped away. Residual misalignment after the
warp is the angle difference itself; it is NOT a defect of the alignment
and must not be claimed as one.

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

Trust gates: a homography EXISTS for any 4 correspondences — 4 points
determine it exactly — so a low match or inlier count is not evidence the
alignment is real. A real-world angulation pair produced 16 Lowe matches
and only 5 RANSAC inliers, and the fitted matrix was degenerate: its line
at infinity crossed the image, so ``warpPerspective`` folded the whole
radiograph through a point into a useless fan. Two defenses guard against
this: :data:`_MIN_ALIGNMENT_INLIERS` is a trust floor (not the existence
minimum) applied both to Lowe matches before fitting and to RANSAC
inliers after, and :func:`validate_alignment_homography` refuses
geometrically impossible warps (line at infinity crossing the frame,
reflections, non-convex folds, implausible scale) before any pixel is
warped. A refused alignment raises :class:`PivotAlignmentError`; the GUI
falls back to side-by-side display, which is the correct presentation of
a pair we cannot trustworthily register.

Single source of truth: ``canvas_a``, ``canvas_b``, ``canvas_offset``,
``inlier_count``, and ``mean_alignment_error_px`` on
:class:`PivotAlignment` are computed here once and returned to callers
verbatim. A GUI must not recompute them from the images or the
homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.preprocessing import preprocess_for_matching

_LOWE_RATIO = 0.75
_RANSAC_REPROJ_THRESHOLD = 3.0

# Provisional trust floor for the alignment, in correspondences. 4 is the
# homography existence minimum and any 4 correspondences fit a homography
# exactly, so counts near 4 carry no evidence the alignment is real. 10 is
# chosen from real-pair experience: a usable real angulation pair aligned
# with 12 RANSAC inliers, while a garbage degenerate one had 5. Update by
# experiment, not by guess.
_MIN_ALIGNMENT_INLIERS = 10

# Bounds for the warped-area sanity check in validate_alignment_homography.
# Same-sensor radiograph pairs are near scale 1; a warp that shrinks or
# blows up the image by more than 10x is not a credible overlap alignment.
_AREA_RATIO_MIN = 0.1
_AREA_RATIO_MAX = 10.0

# Numerical floors: |H[2,2]| below this cannot be normalized; a corner
# homogeneous w at or below this is on (or past) the line at infinity.
_H22_EPS = 1e-8
_CORNER_W_EPS = 1e-6

# Defensive memory guard for the union canvas. A validator-sane homography
# (finite, positive corner w, source winding kept, area ratio near 1) can
# still carry a huge translation, in which case the union bounding box of
# the two radiographs explodes even though each image alone is small. 16x
# the larger input area allows any plausible overlap layout (side by side,
# diagonal, tall) while refusing canvases that would be almost entirely
# empty black.
_MAX_CANVAS_AREA_RATIO = 16.0


class PivotAlignmentError(Exception):
    """Raised when a trustworthy rough overlap alignment cannot be fit.

    Triggered if SIFT cannot find enough features, Lowe-filtered matches
    or RANSAC inliers fall below the :data:`_MIN_ALIGNMENT_INLIERS` trust
    floor, RANSAC fails outright, or the fitted homography fails
    :func:`validate_alignment_homography`. The pivot viewer should fall
    back to a plain side-by-side display on this error and surface the
    message.
    """


@dataclass(frozen=True)
class PivotAlignment:
    """Union-canvas pair built for cross-fade comparison display.

    ``canvas_a`` and ``canvas_b`` are the SAME size: the union bounding
    box of image_a and of image_b mapped through the alignment homography.
    ``canvas_a`` holds image_a placed unresampled at its offset;
    ``canvas_b`` holds image_b warped by the translation-compensated
    homography. Regions covered by neither radiograph are zero (black).
    At slider 0 a viewer shows A in place, at 1 the warped B in place,
    and the canvas shows the FULL extent of both — long, wide, whatever
    the union is. ``canvas_offset`` is ``(x_min, y_min)`` of the union
    box expressed in image_a coordinates (image_a sits at row ``-y_min``,
    col ``-x_min`` of the canvas). ``homography`` is the raw b->a matrix,
    NOT the translated one used to render ``canvas_b``.
    """

    canvas_a: np.ndarray
    canvas_b: np.ndarray
    canvas_offset: tuple[int, int]
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


def validate_alignment_homography(
    homography: np.ndarray, image_shape: tuple[int, int]
) -> tuple[bool, str]:
    """Return ``(True, "")`` if warping an image of ``image_shape``
    (rows, cols) by ``homography`` produces a geometrically sane result,
    else ``(False, reason)``.

    A homography can be fit from as few as 4 correspondences, and a bad
    fit can be degenerate in ways RANSAC never notices: if the line at
    infinity of the transform crosses the source frame, the homogeneous
    w coordinate changes sign across the image and ``warpPerspective``
    folds the whole radiograph through a point — the output is a "fan",
    not a registered image. This validator refuses such matrices before
    any pixel is warped. Checks, in order:

    1. Finite 3x3 with ``|H[2,2]| > 1e-8`` (normalizable).
    2. Corner positivity: the homogeneous w of all four mapped source
       corners must be strictly positive — a sign change or near-zero
       means the line at infinity crosses the image (the fold/fan case).
    3. Orientation/convexity: the mapped corner quadrilateral must keep
       the source winding and stay convex (no reflection, no
       self-intersection).
    4. Area ratio: the mapped quadrilateral area over the source area
       must be within ``[0.1, 10.0]`` — same-sensor radiograph pairs are
       near scale 1, so anything far outside is not a credible alignment.

    Pure function, no side effects; safe to unit-test in isolation.
    """
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        shape = getattr(homography, "shape", None)
        return False, f"homography must be a 3x3 matrix, got shape {shape}"
    h_arr = homography.astype(np.float64, copy=False)
    if not np.all(np.isfinite(h_arr)):
        return False, "homography contains non-finite values (NaN or inf)"
    if abs(float(h_arr[2, 2])) <= _H22_EPS:
        return False, (
            f"homography normalization term H[2,2] is near zero "
            f"({h_arr[2, 2]:.3e}); the matrix cannot be normalized"
        )
    hn = h_arr / h_arr[2, 2]

    rows, cols = int(image_shape[0]), int(image_shape[1])
    if rows <= 0 or cols <= 0:
        return False, f"image_shape must be positive, got {image_shape}"
    w, h = float(cols), float(rows)

    # Source corners walked in a fixed winding; the same order is used by
    # the orientation check below so winding comparisons are meaningful.
    corners = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))

    corner_ws = [hn[2, 0] * x + hn[2, 1] * y + 1.0 for x, y in corners]
    if any(wi <= _CORNER_W_EPS for wi in corner_ws):
        rounded = [round(float(v), 3) for v in corner_ws]
        return False, (
            f"corner homogeneous w values {rounded} include a non-positive "
            f"term: the line at infinity crosses the image and the warp "
            f"folds the radiograph through a point"
        )

    mapped = []
    for (x, y), wi in zip(corners, corner_ws):
        mx = (hn[0, 0] * x + hn[0, 1] * y + hn[0, 2]) / wi
        my = (hn[1, 0] * x + hn[1, 1] * y + hn[1, 2]) / wi
        mapped.append((mx, my))

    for i in range(4):
        x0, y0 = mapped[i]
        x1, y1 = mapped[(i + 1) % 4]
        x2, y2 = mapped[(i + 2) % 4]
        cross_z = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if cross_z <= 0.0:
            return False, (
                "mapped corners are reflected or non-convex (winding flips "
                "against the source); the warp mirrors or self-intersects "
                "the radiograph"
            )

    shoelace = 0.0
    for i in range(4):
        x0, y0 = mapped[i]
        x1, y1 = mapped[(i + 1) % 4]
        shoelace += x0 * y1 - x1 * y0
    area_ratio = abs(shoelace) / 2.0 / (w * h)
    if not (_AREA_RATIO_MIN <= area_ratio <= _AREA_RATIO_MAX):
        return False, (
            f"mapped area ratio {area_ratio:.4f} is outside "
            f"[{_AREA_RATIO_MIN}, {_AREA_RATIO_MAX}]; a same-sensor "
            f"radiograph pair should warp near scale 1"
        )

    return True, ""


def _build_union_canvases(
    image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Build the same-size union-canvas pair for cross-fade display.

    Mirrors the corner-union math of ``extension_stitch._compose``
    (reimplemented here on purpose — extension_stitch.py is not modified
    or imported for this): map image_b's corners through the homography,
    take the union bounding box with image_a's own corners, and render
    both images into a canvas of that size. image_a is placed by exact
    slice assignment (no resampling); image_b is warped once by the
    translation-compensated homography.

    Returns ``(canvas_a, canvas_b, (x_min, y_min))``.

    Raises:
        PivotAlignmentError: the union canvas would exceed
            :data:`_MAX_CANVAS_AREA_RATIO` times the larger input area
            (the pair barely overlaps; a mostly-black giant canvas is not
            a useful comparison view).
    """
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape

    corners_a = np.float32(
        [[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]
    ).reshape(-1, 1, 2)
    corners_b = np.float32(
        [[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]
    ).reshape(-1, 1, 2)
    warped_b_corners = cv2.perspectiveTransform(corners_b, homography)

    all_corners = np.concatenate([corners_a, warped_b_corners], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    canvas_w = int(x_max - x_min)
    canvas_h = int(y_max - y_min)

    if canvas_h * canvas_w > _MAX_CANVAS_AREA_RATIO * max(h_a * w_a, h_b * w_b):
        raise PivotAlignmentError(
            "The two radiographs barely overlap and the combined view "
            "would be implausibly large - showing the two angles side by "
            "side instead."
        )

    translation = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    canvas_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_a[-y_min : -y_min + h_a, -x_min : -x_min + w_a] = image_a

    canvas_b = cv2.warpPerspective(
        image_b, translation @ homography, (canvas_w, canvas_h)
    )

    return canvas_a, canvas_b, (int(x_min), int(y_min))


def align_for_pivot(image_a: np.ndarray, image_b: np.ndarray) -> PivotAlignment:
    """Roughly register ``image_b`` onto ``image_a`` for comparison viewing.

    Both inputs are uint8 2D grayscale arrays. They are passed through
    :func:`preprocess_for_matching` with CLAHE on, then SIFT + Lowe +
    RANSAC fits a homography mapping ``image_b`` -> ``image_a``. Both
    images are then rendered onto a shared union canvas sized to the full
    extent of both (corner union of image_a and the mapped image_b, the
    same construction the extension stitcher uses), so nothing from
    either radiograph is cropped away. The returned
    ``mean_alignment_error_px`` is the mean reprojection error of the
    RANSAC inliers and is the honesty signal a UI should show alongside
    the viewer.

    This is a comparison-view registration of two different-angle real
    radiographs. Residual misalignment after the warp IS the parallax of
    the angle difference. It is NOT a stitch, NOT an angulation correction,
    NOT a 3D reconstruction.

    Raises:
        ValueError: invalid inputs (non-2D, empty, or mismatched dtypes).
        PivotAlignmentError: not enough matchable features to clear the
            trust floor (:data:`_MIN_ALIGNMENT_INLIERS` Lowe matches before
            fitting and as many RANSAC inliers after), RANSAC failure, a
            fitted homography that fails
            :func:`validate_alignment_homography` (degenerate geometry),
            or a union canvas exceeding :data:`_MAX_CANVAS_AREA_RATIO`
            times the larger input (the pair barely overlaps).
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

    if len(good) < _MIN_ALIGNMENT_INLIERS:
        raise PivotAlignmentError(
            f"Only {len(good)} aligned features between these two "
            f"radiographs - too few to trust an overlap alignment. "
            f"Showing the two angles side by side instead."
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
    if inlier_count < _MIN_ALIGNMENT_INLIERS:
        raise PivotAlignmentError(
            f"Only {inlier_count} aligned features between these two "
            f"radiographs - too few to trust an overlap alignment. "
            f"Showing the two angles side by side instead."
        )

    ok, reason = validate_alignment_homography(homography, image_b.shape)
    if not ok:
        raise PivotAlignmentError(
            f"The alignment between these two radiographs is not "
            f"geometrically trustworthy ({reason}) - showing the two "
            f"angles side by side instead."
        )

    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    projected = cv2.perspectiveTransform(src_inliers, homography)
    errors = np.linalg.norm(projected - dst_inliers, axis=2).ravel()
    mean_alignment_error_px = float(errors.mean())

    canvas_a, canvas_b, canvas_offset = _build_union_canvases(
        image_a, image_b, homography
    )

    return PivotAlignment(
        canvas_a=canvas_a,
        canvas_b=canvas_b,
        canvas_offset=canvas_offset,
        inlier_count=inlier_count,
        mean_alignment_error_px=mean_alignment_error_px,
        homography=homography,
    )
