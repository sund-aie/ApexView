"""Characterization experiment: extension vs angulation reprojection error.

This is a research SCRIPT, not part of the shipped engine. It calls the real
`stitch_extension` engine on procedurally generated grayscale scenes warped
under two distinct distortion regimes:

  * EXTENSION regime: a pure planar shift + tiny in-plane rotation applied to
    the rendered scene (the case stitch_extension is designed for). A single
    homography exactly relates image_a and image_b.

  * ANGULATION regime: a TRUE out-of-plane camera tilt of a multi-depth 3D
    scene. The scene has a checkerboard "background" layer and a blob/rect
    "foreground" layer at different depths, straddling the rotation axis.
    Rotating the camera about a vertical axis produces depth-dependent
    parallax: near features shift more than far features. NO single 2D
    homography can model this; a homography fit must accept high reprojection
    error or sacrifice inliers.

The output is a printed text table of mean reprojection error AND inlier
counts across multiple random seeds, plus a plain-language summary that
suggests where the two regimes stop overlapping on either signal. NO
threshold is written into the engine here — that is a separate,
human-reviewed decision.
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
FOCAL = float(IMG_W)
BG_DEPTH = 3.0 * FOCAL
FG_DEPTH = 1.5 * FOCAL
AXIS_DEPTH = 2.0 * FOCAL

SEEDS = [11, 23, 47, 89, 137]
ANGULATION_ANGLES = [2, 5, 10, 15, 20, 25, 30, 40]
EXTENSION_ANGLES = [0, 1, 2, 3]


def _make_background(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)
    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 40
            else:
                img[y : y + tile, x : x + tile] = 215
    rng = np.random.default_rng(seed * 31 + 7)
    for _ in range(20):
        cx = int(rng.integers(5, w - 5))
        cy = int(rng.integers(5, h - 5))
        cv2.circle(img, (cx, cy), 2, int(rng.integers(0, 256)), -1)
    return img


def _make_foreground(
    seed: int, h: int = IMG_H, w: int = IMG_W
) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        cx = int(rng.integers(10, w - 10))
        cy = int(rng.integers(10, h - 10))
        radius = int(rng.integers(3, 8))
        color = int(rng.integers(0, 256))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    for _ in range(40):
        x1 = int(rng.integers(5, w - 25))
        y1 = int(rng.integers(5, h - 25))
        side = int(rng.integers(6, 18))
        color = int(rng.integers(0, 256))
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color, -1)
        cv2.rectangle(mask, (x1, y1), (x1 + side, y1 + side), 255, -1)
    return img, mask


def _layer_homography(
    angle_deg: float,
    layer_z: float,
    w: int = IMG_W,
    h: int = IMG_H,
    focal: float = FOCAL,
    axis_z: float = AXIS_DEPTH,
) -> np.ndarray:
    """Pixel→pixel homography for a flat layer at depth ``layer_z`` when the
    scene rotates about a vertical axis at depth ``axis_z`` by ``angle_deg``,
    viewed by a pinhole camera with the given ``focal`` length.

    Because the rotation axis is at a DIFFERENT depth than the layer, the
    rotation induces a camera-relative lateral translation as well as a
    rotation. That translation makes the resulting per-layer homographies
    differ across depths — which is the parallax we want to characterize.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = []
    for u, v in src:
        X = (u - cx) * layer_z / focal
        Y = (v - cy) * layer_z / focal
        Z = layer_z
        Zr = Z - axis_z
        Xn = X * c + Zr * s
        Zn = -X * s + Zr * c
        Zf = Zn + axis_z
        if Zf <= 1e-3:
            return np.eye(3, dtype=np.float64)
        un = cx + focal * Xn / Zf
        vn = cy + focal * Y / Zf
        dst.append([un, vn])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _render_two_layer_scene(seed: int, angle_deg: float) -> np.ndarray:
    """Render the two-layer 3D scene from a camera tilted by ``angle_deg``."""
    bg = _make_background(seed)
    fg, mask = _make_foreground(seed)
    h_bg = _layer_homography(angle_deg, BG_DEPTH)
    h_fg = _layer_homography(angle_deg, FG_DEPTH)
    warped_bg = cv2.warpPerspective(bg, h_bg, (IMG_W, IMG_H))
    warped_fg = cv2.warpPerspective(fg, h_fg, (IMG_W, IMG_H))
    warped_mask = cv2.warpPerspective(mask, h_fg, (IMG_W, IMG_H))
    out = warped_bg.copy()
    out[warped_mask > 127] = warped_fg[warped_mask > 127]
    return out


def _extension_homography(
    w: int, h: int, angle_deg: float, dx: float, dy: float = 0.0
) -> np.ndarray:
    """Translation + small in-plane rotation, no perspective."""
    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = rot
    homography[0, 2] += dx
    homography[1, 2] += dy
    return homography


@dataclass
class Trial:
    regime: str
    angle_deg: float
    seed: int
    error: float | None
    inliers: int | None
    failure: str | None


def _run_trial(regime: str, angle_deg: float, seed: int) -> Trial:
    image_a = _render_two_layer_scene(seed, 0.0)
    if regime == "extension":
        rng = np.random.default_rng(seed ^ 0xA5A5)
        dx = float(rng.uniform(40.0, 90.0))
        homography = _extension_homography(IMG_W, IMG_H, angle_deg, dx)
        image_b = cv2.warpPerspective(image_a, homography, (IMG_W, IMG_H))
    elif regime == "angulation":
        image_b = _render_two_layer_scene(seed, angle_deg)
    else:
        raise ValueError(f"unknown regime: {regime}")

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
        return "INSUF"
    return f"{t.error:7.3f}"


def _fmt_inl(t: Trial) -> str:
    if t.failure is not None:
        return "INSUF"
    return f"{t.inliers:>5d}"


def _err_stats(trials: list[Trial]) -> tuple[str, str, str]:
    errs = [t.error for t in trials if t.error is not None]
    if not errs:
        return ("--", "--", "--")
    return f"{min(errs):.3f}", f"{statistics.median(errs):.3f}", f"{max(errs):.3f}"


def _inl_stats(trials: list[Trial]) -> tuple[str, str, str]:
    inls = [t.inliers for t in trials if t.inliers is not None]
    if not inls:
        return ("--", "--", "--")
    return f"{min(inls)}", f"{int(statistics.median(inls))}", f"{max(inls)}"


def _print_table(title: str, rows, formatter, stats_fn) -> None:
    print("=" * 108)
    print(title)
    print("=" * 108)
    seed_hdr = " ".join(f"s={s:<5}" for s in SEEDS)
    print(f"{'regime':<11} {'angle':>6}  {seed_hdr}  {'min':>7} {'med':>7} {'max':>7}")
    print("-" * 108)
    for regime, angle, trials in rows:
        cells = " ".join(f"{formatter(t):>7}" for t in trials)
        mn, md, mx = stats_fn(trials)
        print(f"{regime:<11} {angle:>6.1f}  {cells}  {mn:>7} {md:>7} {mx:>7}")


def main() -> None:
    rows: list[tuple[str, float, list[Trial]]] = []
    for angle in EXTENSION_ANGLES:
        trials = [_run_trial("extension", float(angle), s) for s in SEEDS]
        rows.append(("extension", float(angle), trials))
    for angle in ANGULATION_ANGLES:
        trials = [_run_trial("angulation", float(angle), s) for s in SEEDS]
        rows.append(("angulation", float(angle), trials))

    _print_table("REPROJECTION ERROR (pixels)", rows, _fmt_err, _err_stats)
    print()
    _print_table("INLIER COUNT", rows, _fmt_inl, _inl_stats)

    ext_errs: list[float] = []
    ext_inls: list[int] = []
    ang_err: dict[float, list[float]] = {}
    ang_inl: dict[float, list[int]] = {}
    ang_fail: dict[float, int] = {}
    for regime, angle, trials in rows:
        if regime == "extension":
            ext_errs.extend(t.error for t in trials if t.error is not None)
            ext_inls.extend(t.inliers for t in trials if t.inliers is not None)
        else:
            ang_err[angle] = [t.error for t in trials if t.error is not None]
            ang_inl[angle] = [t.inliers for t in trials if t.inliers is not None]
            ang_fail[angle] = sum(1 for t in trials if t.failure is not None)

    print()
    print("SUMMARY")
    print("-------")
    if ext_errs:
        print(
            f"EXTENSION regime (planar warp of two-layer composite):\n"
            f"  reprojection error range = [{min(ext_errs):.3f}, "
            f"{max(ext_errs):.3f}] px, median {statistics.median(ext_errs):.3f}\n"
            f"  inlier count range       = [{min(ext_inls)}, "
            f"{max(ext_inls)}], median {int(statistics.median(ext_inls))}"
        )
    print()
    print("ANGULATION regime (true 3D camera tilt of multi-depth scene):")
    for angle in ANGULATION_ANGLES:
        errs = ang_err.get(float(angle), [])
        inls = ang_inl.get(float(angle), [])
        fails = ang_fail.get(float(angle), 0)
        bits = []
        if errs:
            bits.append(
                f"err [{min(errs):.3f}, {max(errs):.3f}] med {statistics.median(errs):.3f} px"
            )
        if inls:
            bits.append(
                f"inl [{min(inls)}, {max(inls)}] med {int(statistics.median(inls))}"
            )
        if fails:
            bits.append(f"InsufficientOverlap: {fails}/{len(SEEDS)}")
        suffix = "; ".join(bits) if bits else "no data"
        print(f"  {angle:>4.1f} deg: {suffix}")

    print()
    print("SIGNAL SEPARATION")
    print("-----------------")
    ext_err_max = max(ext_errs) if ext_errs else float("nan")
    ext_inl_min = min(ext_inls) if ext_inls else None

    err_sep: float | None = None
    inl_sep: float | None = None
    for angle in ANGULATION_ANGLES:
        errs = ang_err.get(float(angle), [])
        inls = ang_inl.get(float(angle), [])
        fails = ang_fail.get(float(angle), 0)
        if err_sep is None:
            if (errs and min(errs) > ext_err_max) or (fails == len(SEEDS)):
                err_sep = float(angle)
        if inl_sep is None and ext_inl_min is not None:
            if (inls and max(inls) < ext_inl_min) or (fails == len(SEEDS)):
                inl_sep = float(angle)

    if err_sep is not None:
        gap_at_sep = min(ang_err.get(err_sep, [float("nan")])) - ext_err_max
        print(
            f"  By reprojection error: the smallest angle at which every angulation "
            f"trial exceeds the worst extension trial ({ext_err_max:.3f} px) is "
            f"{err_sep:.1f} deg. Gap at that angle = {gap_at_sep:.3f} px."
        )
    else:
        print("  By reprojection error: no clean separation in swept range.")

    if inl_sep is not None:
        gap_at_inl = ext_inl_min - max(ang_inl.get(inl_sep, [0]))
        print(
            f"  By inlier count:       the smallest angle at which every angulation "
            f"trial has fewer inliers than the worst extension trial ({ext_inl_min}) "
            f"is {inl_sep:.1f} deg. Gap at that angle = {gap_at_inl} inliers."
        )
    elif ext_inl_min is None:
        print("  By inlier count: no extension inlier data.")
    else:
        print("  By inlier count: no clean separation in swept range.")

    print()
    print("NOTE: this is data for a human to read. No threshold has been "
          "written into the engine.")


if __name__ == "__main__":
    main()
