"""Characterization experiment: extension vs angulation reprojection error.

This is a research SCRIPT, not part of the shipped engine. It calls the real
`stitch_extension` engine on procedurally generated grayscale images warped
under two distinct distortion regimes:

  * EXTENSION regime: pure planar shift + tiny in-plane rotation
    (the case stitch_extension is designed for).
  * ANGULATION regime: a perspective warp that simulates viewing the same
    teeth from a different beam angle (the case a future angulation engine
    will handle, and the case extension stitching should NOT silently
    accept).

The output is a printed text table of mean reprojection error and inlier
counts across multiple random seeds, plus a plain-language summary that
suggests where the two regimes stop overlapping. NO threshold is written
into the engine here — that is a separate, human-reviewed decision.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import cv2
import numpy as np

from apexview.engine.extension_stitch import (
    InsufficientOverlapError,
    stitch_extension,
)

IMG_H = 240
IMG_W = 320
SEEDS = [11, 23, 47, 89, 137]
ANGULATION_ANGLES = [0, 5, 10, 15, 20, 25, 30, 40]
EXTENSION_ANGLES = [0, 1, 2, 3]


def _make_feature_image(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Procedural high-contrast grayscale image, same recipe as the engine tests."""
    img = np.full((h, w), 128, dtype=np.uint8)
    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 40
            else:
                img[y : y + tile, x : x + tile] = 215

    rng = np.random.default_rng(seed)
    for _ in range(80):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        radius = int(rng.integers(3, 8))
        color = int(rng.integers(0, 256))
        cv2.circle(img, (cx, cy), radius, color, thickness=-1)
    for _ in range(40):
        x1 = int(rng.integers(5, w - 25))
        y1 = int(rng.integers(5, h - 25))
        side = int(rng.integers(6, 18))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color, thickness=-1)
    return img


def _extension_homography(
    w: int, h: int, angle_deg: float, dx: float, dy: float = 0.0
) -> np.ndarray:
    """Translation + small in-plane rotation about the image center.

    Pure planar: no perspective terms. This is what a sensor sliding sideways
    at the same beam angle is expected to produce.
    """
    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


def _angulation_homography(w: int, h: int, angle_deg: float) -> np.ndarray:
    """Perspective warp simulating out-of-plane rotation of the image plane.

    The image plane is tilted about the vertical axis through its center by
    ``angle_deg`` and re-projected through a pinhole camera with focal length
    ~ image width. This produces a genuine homography with non-zero
    perspective terms, NOT an affine map.
    """
    f = float(w)
    theta = math.radians(angle_deg)
    cx, cy = w / 2.0, h / 2.0
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = []
    for x, y in src:
        ox = x - cx
        oy = y - cy
        new_ox = ox * math.cos(theta)
        depth = ox * math.sin(theta)
        denom = f + depth
        scale = f / denom if denom > 1e-6 else 1.0
        dst.append([cx + new_ox * scale, cy + oy * scale])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


@dataclass
class Trial:
    regime: str
    angle_deg: float
    seed: int
    error: float | None
    inliers: int | None
    failure: str | None


def _run_trial(regime: str, angle_deg: float, seed: int) -> Trial:
    image_a = _make_feature_image(seed)
    if regime == "extension":
        rng = np.random.default_rng(seed ^ 0xA5A5)
        dx = float(rng.uniform(40.0, 90.0))
        homography = _extension_homography(IMG_W, IMG_H, angle_deg, dx)
    elif regime == "angulation":
        homography = _angulation_homography(IMG_W, IMG_H, angle_deg)
    else:
        raise ValueError(f"unknown regime: {regime}")

    image_b = cv2.warpPerspective(image_a, homography, (IMG_W, IMG_H))

    try:
        result = stitch_extension(image_a, image_b)
    except InsufficientOverlapError as exc:
        return Trial(regime, angle_deg, seed, None, None, str(exc))
    return Trial(
        regime, angle_deg, seed,
        float(result.mean_reprojection_error),
        int(result.inlier_count),
        None,
    )


def _fmt_err(t: Trial) -> str:
    if t.failure is not None:
        return "INSUFFICIENT_OVERLAP"
    return f"{t.error:8.3f}"


def _aggregate(trials: list[Trial]) -> tuple[str, str, str, str]:
    """Return (min, median, max, median_inliers) as formatted strings."""
    errs = [t.error for t in trials if t.error is not None]
    inls = [t.inliers for t in trials if t.inliers is not None]
    if not errs:
        return ("--", "--", "--", "--")
    return (
        f"{min(errs):.3f}",
        f"{statistics.median(errs):.3f}",
        f"{max(errs):.3f}",
        f"{int(statistics.median(inls))}",
    )


def main() -> None:
    rows: list[tuple[str, float, list[Trial]]] = []

    for angle in EXTENSION_ANGLES:
        trials = [_run_trial("extension", float(angle), s) for s in SEEDS]
        rows.append(("extension", float(angle), trials))

    for angle in ANGULATION_ANGLES:
        trials = [_run_trial("angulation", float(angle), s) for s in SEEDS]
        rows.append(("angulation", float(angle), trials))

    header_seeds = " ".join(f"seed={s:>5}" for s in SEEDS)
    print(
        f"{'regime':<11} {'angle':>6}  {header_seeds}  "
        f"{'min':>8} {'median':>8} {'max':>8}  {'med_inl':>7}"
    )
    print("-" * (11 + 1 + 6 + 2 + len(header_seeds) + 2 + 8 + 1 + 8 + 1 + 8 + 2 + 7))

    for regime, angle, trials in rows:
        cells = " ".join(f"{_fmt_err(t):>10}" for t in trials)
        mn, md, mx, mi = _aggregate(trials)
        print(
            f"{regime:<11} {angle:>6.1f}  {cells}  "
            f"{mn:>8} {md:>8} {mx:>8}  {mi:>7}"
        )

    print()
    print("SUMMARY")
    print("-------")

    ext_errs: list[float] = []
    for regime, _, trials in rows:
        if regime != "extension":
            continue
        for t in trials:
            if t.error is not None:
                ext_errs.append(t.error)
    if ext_errs:
        print(
            f"EXTENSION regime: error range across all seeds and small "
            f"in-plane rotations = [{min(ext_errs):.3f}, {max(ext_errs):.3f}] px; "
            f"median = {statistics.median(ext_errs):.3f} px."
        )
    else:
        print("EXTENSION regime produced no successful trials (unexpected).")

    print()
    print("ANGULATION regime (per simulated tilt angle):")
    angulation_by_angle: dict[float, list[float]] = {}
    angulation_failures: dict[float, int] = {}
    for regime, angle, trials in rows:
        if regime != "angulation":
            continue
        angulation_by_angle[angle] = [t.error for t in trials if t.error is not None]
        angulation_failures[angle] = sum(1 for t in trials if t.failure is not None)
        successes = angulation_by_angle[angle]
        fails = angulation_failures[angle]
        if successes:
            print(
                f"  {angle:>5.1f} deg: error range "
                f"[{min(successes):.3f}, {max(successes):.3f}] px, "
                f"median {statistics.median(successes):.3f} px"
                + (f", failures: {fails}/{len(SEEDS)}" if fails else "")
            )
        else:
            print(f"  {angle:>5.1f} deg: ALL seeds raised InsufficientOverlapError")

    print()
    ext_max = max(ext_errs) if ext_errs else float("nan")
    separation_angle: float | None = None
    for angle in ANGULATION_ANGLES:
        succ = angulation_by_angle.get(float(angle), [])
        fails = angulation_failures.get(float(angle), 0)
        if (succ and min(succ) > ext_max) or (fails == len(SEEDS)):
            separation_angle = float(angle)
            break

    if separation_angle is None:
        print(
            "No angulation angle in the swept range cleanly separates from the "
            "extension regime's worst-case error. Either widen the sweep or "
            "accept overlap in the chosen threshold."
        )
    else:
        print(
            f"Separation point: at {separation_angle:.1f} deg of simulated "
            f"angulation, the minimum observed angulation error (or unanimous "
            f"InsufficientOverlapError) exceeds the EXTENSION regime's worst "
            f"case of {ext_max:.3f} px. This is the smallest angle at which "
            f"the two regimes stop overlapping in this experiment."
        )

    print()
    print(
        "NOTE: this is data for a human to read. No threshold has been "
        "written into the engine."
    )


if __name__ == "__main__":
    main()
