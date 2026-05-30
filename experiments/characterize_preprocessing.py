"""Characterization experiment: does preprocessing real radiographs make two-view
geometry recoverable?

This is a research SCRIPT, not a shipped engine feature. It calls the real
engine (`apexview.io.dicom_reader.load_dicom`, the SIFT matching convention
mirrored from `apexview.engine.extension_stitch`, and
`apexview.engine.stereo_geometry.estimate_fundamental_from_points` /
`mean_symmetric_epipolar_error`) on PAIRS of real DICOMs the user supplies on
their local machine. Nothing is copied, embedded, or committed back to the
repo — only this script lives in the repo. The user runs it locally.

WHAT WE MEASURE (and why):

The metric that matters is SURVIVING GEOMETRY INLIERS and how spatially
spread out they are, NOT raw keypoint count. More features are worthless if
they don't correctly correspond across the two images, and even many true
matches that are clustered into a small region can't pin down a stable
fundamental matrix. The script reports keypoint count as a vanity metric
(clearly labelled), Lowe-filtered match count, then the primary outcomes:
whether `estimate_two_view_geometry` accepts the pair at all, and if so its
inlier count, mean epipolar error, and a 4x4-grid spatial coverage of the
surviving inliers.

NO preprocessing parameter was tuned to flatter any result. CLAHE uses
clipLimit=2.0, tileGridSize=(8,8) — standard radiograph defaults — and the
parameters are printed in the output so any reader can see what was tried.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from apexview.engine.stereo_geometry import (
    InsufficientGeometryError,
    estimate_fundamental_from_points,
    mean_symmetric_epipolar_error,
)
from apexview.io.dicom_reader import (
    InvalidDicomError,
    RadiographImage,
    load_dicom,
)

# Matching convention mirrored from extension_stitch.py / stereo_geometry.py
_LOWE_RATIO = 0.75
_MIN_FOR_F = 8  # 8-point algorithm floor
_GRID_N = 4  # 4x4 spatial-coverage grid

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

VARIANTS = ("raw", "clahe", "equalize")


@dataclass
class VariantResult:
    variant: str
    kp_a: int
    kp_b: int
    good_matches: int
    success: bool
    failure_reason: str = ""
    inlier_count: int = 0
    mean_epipolar_error: float = float("nan")
    coverage_a: int = 0  # inlier cells used in image_a (0..16)
    coverage_b: int = 0  # inlier cells used in image_b (0..16)


@dataclass
class PairResult:
    pair_label: str
    file_a: str
    file_b: str
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    variants: list[VariantResult] = field(default_factory=list)


def preprocess(image_u8: np.ndarray, variant: str) -> np.ndarray:
    """Apply a named preprocessing variant to an 8-bit grayscale image."""
    if image_u8.dtype != np.uint8 or image_u8.ndim != 2:
        raise ValueError("preprocess expects a 2D uint8 image")
    if variant == "raw":
        return image_u8
    if variant == "clahe":
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(image_u8)
    if variant == "equalize":
        return cv2.equalizeHist(image_u8)
    raise ValueError(f"unknown preprocessing variant: {variant!r}")


def _match_points(
    image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """SIFT + Lowe-ratio match mirroring the engine; returns
    ``(pts_a, pts_b, kp_a_count, kp_b_count)``.
    """
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy(), len(kp_a), len(kp_b)

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
    return pts_a, pts_b, len(kp_a), len(kp_b)


def grid_coverage(pts: np.ndarray, h: int, w: int, n: int = _GRID_N) -> int:
    """Return number of distinct cells in an n x n grid covering [0,w)x[0,h)
    that contain at least one of the given points. Clipped to grid bounds.
    """
    if pts.size == 0:
        return 0
    xs = np.clip((pts[:, 0] / max(w, 1) * n).astype(int), 0, n - 1)
    ys = np.clip((pts[:, 1] / max(h, 1) * n).astype(int), 0, n - 1)
    cells = np.unique(ys * n + xs)
    return int(cells.size)


def analyze_pair(image_a_u8: np.ndarray, image_b_u8: np.ndarray) -> list[VariantResult]:
    """Run all preprocessing variants on a pair and return per-variant metrics.

    Used by the script's main loop and by the unit tests; intentionally pure
    (no I/O, no printing) so it can be exercised on synthetic inputs.
    """
    out: list[VariantResult] = []
    h_a, w_a = image_a_u8.shape
    h_b, w_b = image_b_u8.shape

    for variant in VARIANTS:
        a_proc = preprocess(image_a_u8, variant)
        b_proc = preprocess(image_b_u8, variant)
        pts_a, pts_b, kp_a, kp_b = _match_points(a_proc, b_proc)
        result = VariantResult(
            variant=variant,
            kp_a=kp_a,
            kp_b=kp_b,
            good_matches=int(pts_a.shape[0]),
            success=False,
        )

        if pts_a.shape[0] < _MIN_FOR_F:
            result.failure_reason = (
                f"only {pts_a.shape[0]} good matches; below 8-point floor"
            )
            out.append(result)
            continue

        try:
            F, mask = estimate_fundamental_from_points(pts_a, pts_b)
        except InsufficientGeometryError as exc:
            result.failure_reason = str(exc)
            out.append(result)
            continue

        inlier_pts_a = pts_a[mask]
        inlier_pts_b = pts_b[mask]
        result.success = True
        result.inlier_count = int(mask.sum())
        result.mean_epipolar_error = mean_symmetric_epipolar_error(
            F, inlier_pts_a, inlier_pts_b
        )
        result.coverage_a = grid_coverage(inlier_pts_a, h_a, w_a)
        result.coverage_b = grid_coverage(inlier_pts_b, h_b, w_b)
        out.append(result)

    return out


def _list_dicoms(folder: Path) -> list[Path]:
    """All DICOM-ish files in a folder, sorted by name. Accepts .dcm or no
    extension (DICOM files often have neither)."""
    if not folder.is_dir():
        return []
    entries = [
        p for p in folder.iterdir()
        if p.is_file() and (p.suffix.lower() in {"", ".dcm", ".dicom"})
    ]
    return sorted(entries, key=lambda p: p.name.lower())


def _adjacent_pairs(paths: list[Path]) -> list[tuple[Path, Path]]:
    if len(paths) == 2:
        return [(paths[0], paths[1])]
    return list(zip(paths[:-1], paths[1:]))


def _load_or_error(path: Path) -> RadiographImage | str:
    try:
        return load_dicom(str(path))
    except (FileNotFoundError, InvalidDicomError) as exc:
        return f"  could not load {path.name}: {exc}"


def _print_pair_table(pair: PairResult) -> None:
    print("=" * 100)
    print(f"PAIR: {pair.pair_label}")
    print(f"  A: {pair.file_a}  shape={pair.shape_a}")
    print(f"  B: {pair.file_b}  shape={pair.shape_b}")
    print("-" * 100)
    header = (
        f"{'variant':<9}  {'kp_a':>5}  {'kp_b':>5}  {'good':>5}  "
        f"{'status':<7}  {'inl':>4}  {'err_px':>7}  "
        f"{'cov_A':>6}  {'cov_B':>6}"
    )
    print(header)
    print("-" * len(header))
    for v in pair.variants:
        if v.success:
            status = "OK"
            inl = f"{v.inlier_count:>4d}"
            err = f"{v.mean_epipolar_error:7.3f}"
            cov_a = f"{v.coverage_a:>2d}/16"
            cov_b = f"{v.coverage_b:>2d}/16"
        else:
            status = "REFUSED"
            inl = "  --"
            err = "    --"
            cov_a = "   --"
            cov_b = "   --"
        print(
            f"{v.variant:<9}  {v.kp_a:>5d}  {v.kp_b:>5d}  {v.good_matches:>5d}  "
            f"{status:<7}  {inl}  {err}  {cov_a}  {cov_b}"
        )
        if not v.success and v.failure_reason:
            print(f"          reason: {v.failure_reason}")
    print()


def _summarize_pair(pair: PairResult) -> str:
    successes = [v for v in pair.variants if v.success]
    if not successes:
        reasons = "; ".join(
            f"{v.variant}: {v.failure_reason or 'refused'}" for v in pair.variants
        )
        return (
            f"  {pair.pair_label}: NO variant produced recoverable geometry. "
            f"Reasons: {reasons}"
        )
    raw_ok = any(v.variant == "raw" and v.success for v in pair.variants)
    best = max(successes, key=lambda v: (v.coverage_a, v.inlier_count))
    spread = "broad" if best.coverage_a >= 10 else (
        "moderate" if best.coverage_a >= 6 else "clustered"
    )
    note = (
        " (raw also succeeded)" if raw_ok
        else " (raw FAILED; preprocessing rescued this pair)"
    )
    return (
        f"  {pair.pair_label}: best variant {best.variant!r} -> "
        f"{best.inlier_count} inliers, coverage {best.coverage_a}/16 ({spread}), "
        f"err {best.mean_epipolar_error:.3f} px{note}"
    )


def _overall_recommendation(pairs: list[PairResult]) -> str:
    if not pairs:
        return "No pairs analyzed."
    n_pairs = len(pairs)
    raw_ok = sum(
        1 for p in pairs if any(v.variant == "raw" and v.success for v in p.variants)
    )
    clahe_ok = sum(
        1 for p in pairs if any(v.variant == "clahe" and v.success for v in p.variants)
    )
    equalize_ok = sum(
        1 for p in pairs
        if any(v.variant == "equalize" and v.success for v in p.variants)
    )
    rescued_by_clahe = sum(
        1
        for p in pairs
        if any(v.variant == "clahe" and v.success for v in p.variants)
        and not any(v.variant == "raw" and v.success for v in p.variants)
    )
    rescued_by_equalize = sum(
        1
        for p in pairs
        if any(v.variant == "equalize" and v.success for v in p.variants)
        and not any(v.variant == "raw" and v.success for v in p.variants)
    )

    # spread quality among successes
    broad_successes = 0
    total_successes = 0
    for p in pairs:
        for v in p.variants:
            if v.success:
                total_successes += 1
                if v.coverage_a >= 10:
                    broad_successes += 1

    lines = [
        f"Pairs analyzed: {n_pairs}",
        f"  raw succeeded:        {raw_ok}/{n_pairs}",
        f"  clahe succeeded:      {clahe_ok}/{n_pairs}  "
        f"(rescued {rescued_by_clahe} pair(s) raw could not handle)",
        f"  equalize succeeded:   {equalize_ok}/{n_pairs}  "
        f"(rescued {rescued_by_equalize} pair(s) raw could not handle)",
    ]
    if total_successes:
        lines.append(
            f"  Of {total_successes} successful runs across all variants, "
            f"{broad_successes} had broad coverage (>= 10/16 cells)."
        )

    if raw_ok == n_pairs:
        verdict = (
            "Raw matching already handled every pair. Preprocessing not "
            "required for THIS dataset; revisit if other pairs fail."
        )
    elif clahe_ok > raw_ok or equalize_ok > raw_ok:
        better = "CLAHE" if clahe_ok >= equalize_ok else "equalize"
        verdict = (
            f"Preprocessing helped: {better} recovered geometry on pairs that "
            f"raw could not. Whether this is reliable for clinical use depends "
            f"on the inlier coverage column above — broad coverage is needed; "
            f"clustered inliers are not trustworthy even when numerous."
        )
    elif clahe_ok == 0 and equalize_ok == 0 and raw_ok == 0:
        verdict = (
            "Honest conclusion: SIFT could not recover two-view geometry on "
            "ANY pair under ANY tested preprocessing. The fundamental matrix "
            "approach is insufficient for these images as-is; a different "
            "feature/matching strategy (or different input quality) is needed."
        )
    else:
        verdict = (
            "Preprocessing did not change the outcome on this dataset. The "
            "successes/failures pattern is the same as raw."
        )
    lines.append("")
    lines.append(f"RECOMMENDATION: {verdict}")
    return "\n".join(lines)


def run_folder(folder: Path) -> int:
    paths = _list_dicoms(folder)
    if len(paths) < 2:
        print(
            f"need at least 2 DICOM files in {folder}; found {len(paths)}",
            file=sys.stderr,
        )
        return 2

    print(f"Folder: {folder}")
    print(f"Files (sorted): {[p.name for p in paths]}")
    print(f"Preprocessing parameters:")
    print(f"  CLAHE: clipLimit={CLAHE_CLIP_LIMIT}, tileGridSize={CLAHE_TILE_GRID}")
    print(f"  equalize: cv2.equalizeHist (global histogram equalization)")
    print()

    pair_results: list[PairResult] = []
    for idx, (pa, pb) in enumerate(_adjacent_pairs(paths), start=1):
        label = f"pair{idx} ({pa.name} + {pb.name})"
        ra = _load_or_error(pa)
        rb = _load_or_error(pb)
        if isinstance(ra, str) or isinstance(rb, str):
            print("=" * 100)
            print(f"PAIR: {label}")
            if isinstance(ra, str):
                print(ra)
            if isinstance(rb, str):
                print(rb)
            print()
            continue
        results = analyze_pair(ra.pixels_u8, rb.pixels_u8)
        pair = PairResult(
            pair_label=label,
            file_a=pa.name,
            file_b=pb.name,
            shape_a=tuple(ra.pixels_u8.shape),
            shape_b=tuple(rb.pixels_u8.shape),
            variants=results,
        )
        pair_results.append(pair)
        _print_pair_table(pair)

    if pair_results:
        print("PER-PAIR SUMMARY")
        print("-" * 100)
        for p in pair_results:
            print(_summarize_pair(p))
        print()
        print("OVERALL")
        print("-" * 100)
        print(_overall_recommendation(pair_results))
    print()
    print(
        "NOTE: data for a human to read. No preprocessing has been baked into "
        "the engine."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="characterize_preprocessing",
        description=(
            "Measure whether preprocessing (CLAHE, equalize) makes two-view "
            "geometry recoverable on real radiograph pairs. The folder is "
            "read locally; no files are copied or committed."
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing two or more DICOM files (read locally).",
    )
    args = parser.parse_args(argv)
    if args.folder is None:
        parser.print_help(sys.stderr)
        return 2
    return run_folder(Path(args.folder))


if __name__ == "__main__":
    raise SystemExit(main())
