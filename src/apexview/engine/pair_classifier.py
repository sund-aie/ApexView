"""Extension vs angulation pair classifier for ApexView.

Thin layer on top of :mod:`apexview.engine.extension_stitch`. Given two
grayscale radiograph candidates from a session, decide whether they are an
EXTENSION pair (same beam angle, sensor translated) that can be stitched, or
an ANGULATION pair (different beam angle) that needs a future
angulation-correction engine.

The decision is keyed on ``StitchResult.inlier_count`` from the existing
stitcher. The Step 1 characterization experiment showed that inlier count is
the clean separator: clean synthetic extensions produced hundreds of RANSAC
inliers, while genuine multi-depth angulation produced at most a few dozen
and at extreme tilt the stitcher raised :class:`InsufficientOverlapError`.

Single source of truth: ``inlier_count``, ``mean_reprojection_error``, and
``stitched_image`` on the returned :class:`ClassificationResult` are READ from
the :class:`StitchResult` produced by the engine — never recomputed here, and
never re-stitched. The classifier is decision logic only. A future
angulation-correction engine will replace the "correction not yet
implemented" branch with a real corrective stitch; the public surface of
:func:`classify_pair` is expected to stay the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from apexview.engine.extension_stitch import (
    InsufficientOverlapError,
    StitchResult,
    stitch_extension,
)

# Tunable. Chosen from the synthetic characterization experiment: clean
# extensions produced inlier counts in [218, 406] while genuine angulation
# topped out at 56 and was usually well under 12. The two regimes did not
# overlap, leaving a wide margin; 100 sits comfortably inside the gap.
# Provisional: real radiographs match less cleanly than synthetic
# checkerboards and this number is expected to come DOWN once we have a
# clinical-image validation set. Update by experiment, not by guess.
EXTENSION_MIN_INLIERS = 100


class PairType(Enum):
    EXTENSION = "extension"
    ANGULATION = "angulation"


@dataclass
class ClassificationResult:
    pair_type: PairType
    inlier_count: int
    mean_reprojection_error: float
    stitched_image: np.ndarray | None
    message: str


def classify_pair(
    image_a: np.ndarray, image_b: np.ndarray
) -> ClassificationResult:
    """Classify a radiograph pair as EXTENSION or ANGULATION.

    Calls the existing :func:`stitch_extension` engine and reads its
    :class:`StitchResult` to make the decision. Bad inputs (``ValueError``
    from the engine's input validation) propagate to the caller untouched —
    a malformed array is a programmer error, not a classification.
    """
    try:
        result: StitchResult = stitch_extension(image_a, image_b)
    except InsufficientOverlapError:
        return ClassificationResult(
            pair_type=PairType.ANGULATION,
            inlier_count=0,
            mean_reprojection_error=math.nan,
            stitched_image=None,
            message=(
                "Matching failed: too few consistent features to fit a planar "
                "homography. Consistent with severe angulation; correction "
                "not yet implemented."
            ),
        )

    if result.inlier_count >= EXTENSION_MIN_INLIERS:
        return ClassificationResult(
            pair_type=PairType.EXTENSION,
            inlier_count=result.inlier_count,
            mean_reprojection_error=result.mean_reprojection_error,
            stitched_image=result.stitched_image,
            message=(
                f"Extension pair: {result.inlier_count} inliers "
                f"(>= {EXTENSION_MIN_INLIERS} threshold); stitched."
            ),
        )

    return ClassificationResult(
        pair_type=PairType.ANGULATION,
        inlier_count=result.inlier_count,
        mean_reprojection_error=result.mean_reprojection_error,
        stitched_image=None,
        message=(
            f"Angulation detected: only {result.inlier_count} inliers "
            f"(< {EXTENSION_MIN_INLIERS} threshold). Correction not yet "
            f"implemented."
        ),
    )
