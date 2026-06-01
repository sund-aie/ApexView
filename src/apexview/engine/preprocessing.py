"""Shared radiograph preprocessing for ApexView.

This is the SINGLE SOURCE OF TRUTH for any pre-SIFT image conditioning the
engine applies. Both :mod:`apexview.engine.extension_stitch` and
:mod:`apexview.engine.stereo_geometry` route their working images through
:func:`preprocess_for_matching` before SIFT runs. No other module should
re-implement CLAHE, do its own histogram tweaking, or otherwise add a second
preprocessing path; if a future consumer (UI, batch job, A/B harness) wants
the "engine recipe", it must call this function.

Chosen parameters and why:

* ``CLAHE_CLIP_LIMIT = 2.0`` and ``CLAHE_TILE_GRID = (8, 8)`` are the
  standard radiograph-enhancement defaults and were validated on real
  intraoral pairs by ``experiments/characterize_preprocessing.py``. The
  measured effect on a real angulation pair was lifting surviving-inlier
  spatial coverage from clustered (~6-7/16 cells) to broad (>=10/16) while
  keeping the RANSAC inlier count comparable. Coverage is the metric that
  matters: broad inlier distributions are needed for stable two-view
  geometry; clustered inliers cannot pin it down even when numerous.

Note for the record (NOT wired into the engine): the same experiment also
evaluated a denser SIFT configuration (``contrastThreshold=0.02`` vs the
default 0.04). It helped on the raw input but, once CLAHE is applied, added
extra keypoints without converting them into broader coverage and produced
clear noise traps on degraded images (>2x baseline keypoints with min-coverage
< 8/16). It is therefore NOT adopted as a default. The constant below
documents the option in case a later evaluation revisits it.
"""

from __future__ import annotations

import cv2
import numpy as np

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Evaluated and NOT adopted as a default. See module docstring.
DENSE_SIFT_CONTRAST_THRESHOLD_NOT_ADOPTED = 0.02


def preprocess_for_matching(
    image_u8: np.ndarray, apply_clahe: bool = True
) -> np.ndarray:
    """Return the image as it should be fed to SIFT.

    With ``apply_clahe=True`` (default), runs CLAHE with the constants above
    and returns a new uint8 array of the same shape. With
    ``apply_clahe=False``, returns the input unchanged (passthrough), useful
    for tests that need to isolate raw-input behavior.

    The caller's input array is never modified in place: CLAHE produces a
    fresh array, and the passthrough branch simply hands the original back.
    """
    if not isinstance(image_u8, np.ndarray):
        raise ValueError("preprocess_for_matching expects a numpy ndarray")
    if image_u8.ndim != 2:
        raise ValueError(
            f"preprocess_for_matching expects a 2D image, got shape {image_u8.shape}"
        )
    if image_u8.dtype != np.uint8:
        raise ValueError(
            f"preprocess_for_matching expects a uint8 image, got dtype {image_u8.dtype}"
        )

    if not apply_clahe:
        return image_u8

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(image_u8)
