"""Characterization experiment: does preprocessing + SIFT-density make two-view
geometry recoverable on real radiographs?

This is a research SCRIPT, not a shipped engine feature. It calls the real
engine (`apexview.io.dicom_reader.load_dicom`, the SIFT matching convention
mirrored from `apexview.engine.extension_stitch`, and
`apexview.engine.stereo_geometry.estimate_fundamental_from_points` /
`mean_symmetric_epipolar_error`) on PAIRS of real DICOMs the user supplies on
their local machine. Nothing is copied, embedded, or committed back to the
repo — only this script lives in the repo. The user runs it locally.

WHAT WE MEASURE (and why):

The metric that matters is SURVIVING RANSAC INLIERS and how spatially spread
out they are, NOT raw keypoint count. More features are worthless if they do
not correctly correspond across the two images, and even many true matches
that are clustered into a small region cannot pin down a stable fundamental
matrix. Move 0 already showed a real case (equalizeHist produced 8631
keypoints but only 7 usable inliers); this Move 1 extends the sweep to also
vary SIFT detection density, with the same honesty constraint: ranking is
coverage-first, and any combination that produces a lot of keypoints but
poor coverage is explicitly flagged as a likely NOISE TRAP.

The sweep is preprocessing x SIFT-density:
  {raw, clahe, equalize} x {sift_default, sift_dense}  ->  6 combinations.

The "dense" SIFT config lowers contrastThreshold only (a single-parameter A/B,
not a parameter hunt); edgeThreshold stays at OpenCV's default. The exact
values are listed in the printed parameter block.

NO parameter was tuned to flatter any result. This script bakes in nothing —
it prints recommendations for a human to read.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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
_BROAD_MIN_COVERAGE = 10  # >= 10/16 cells = "broad" inlier distribution
_NOISE_TRAP_KP_RATIO = 2.0  # combo's keypoints > 2x raw+default baseline
_NOISE_TRAP_MIN_COVERAGE = 8  # ... and min-coverage < 8/16  -> noise trap

# Preprocessing parameters (standard values, stated in output)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# SIFT density parameters (one dense config, single-parameter A/B vs default)
# OpenCV defaults for cv2.SIFT_create: contrastThreshold=0.04, edgeThreshold=10
SIFT_DENSE_CONTRAST_THRESHOLD = 0.02  # lowered from default 0.04
SIFT_DENSE_EDGE_THRESHOLD = 10  # unchanged from default; single-parameter change

PREPROCESS_VARIANTS = ("raw", "clahe", "equalize")
SIFT_CONFIGS = ("sift_default", "sift_dense")

# Combinations in display order: each preprocess variant tried with each SIFT
COMBINATIONS: list[tuple[str, str]] = [
    (p, s) for p in PREPROCESS_VARIANTS for s in SIFT_CONFIGS
]

# Kept for backward-compatibility with earlier test scaffolding.
VARIANTS = PREPROCESS_VARIANTS


@dataclass
class VariantResult:
    variant: str  # combined label, e.g. "clahe+sift_dense"
    preprocess_variant: str
    sift_config: str
    kp_a: int  # VANITY metric — keypoint count, not a quality signal
    kp_b: int  # VANITY metric — keypoint count, not a quality signal
    good_matches: int
    success: bool
    failure_reason: str = ""
    inlier_count: int = 0
    mean_epipolar_error: float = float("nan")
    coverage_a: int = 0  # inlier cells used in image_a (0..16)
    coverage_b: int = 0  # inlier cells used in image_b (0..16)

    @property
    def min_coverage(self) -> int:
        """The conservative spread metric used for ranking: a combo is only
        as good as its less-covered image."""
        return min(self.coverage_a, self.coverage_b)


@dataclass
class PairResult:
    pair_label: str
    file_a: str
    file_b: str
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    variants: list[VariantResult] = field(default_factory=list)

    def baseline_kp_count(self) -> int:
        """Reference keypoint count: raw + sift_default on this pair. Used as
        the denominator for the noise-trap heuristic."""
        for v in self.variants:
            if v.preprocess_variant == "raw" and v.sift_config == "sift_default":
                return max(v.kp_a, v.kp_b)
        return 0


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


def make_sift(sift_config: str):
    """Construct a SIFT detector for a named config.

    "sift_default" uses OpenCV's defaults (matches the engine's stitcher and
    stereo_geometry exactly). "sift_dense" lowers contrastThreshold to admit
    lower-contrast keypoints — useful when radiograph contrast is poor, but
    expected to also raise noise; whether it actually helps geometry is the
    empirical question this experiment asks.
    """
    if sift_config == "sift_default":
        return cv2.SIFT_create()
    if sift_config == "sift_dense":
        return cv2.SIFT_create(
            contrastThreshold=SIFT_DENSE_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_DENSE_EDGE_THRESHOLD,
        )
    raise ValueError(f"unknown sift config: {sift_config!r}")


def _match_points(
    image_a: np.ndarray, image_b: np.ndarray, sift
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """SIFT (configured by caller) + Lowe-ratio match, mirroring the engine's
    matching convention exactly (BFMatcher NORM_L2, knnMatch(des_b, des_a),
    ratio 0.75; queryIdx -> kp_b, trainIdx -> kp_a). Returns
    ``(pts_a, pts_b, kp_a_count, kp_b_count)``.
    """
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
    """Number of distinct cells in an ``n x n`` grid covering ``[0, w) x
    [0, h)`` that contain at least one of the given points.
    """
    if pts.size == 0:
        return 0
    xs = np.clip((pts[:, 0] / max(w, 1) * n).astype(int), 0, n - 1)
    ys = np.clip((pts[:, 1] / max(h, 1) * n).astype(int), 0, n - 1)
    cells = np.unique(ys * n + xs)
    return int(cells.size)


def analyze_pair(image_a_u8: np.ndarray, image_b_u8: np.ndarray) -> list[VariantResult]:
    """Run all preprocess x SIFT-density combinations on a pair and return
    per-combination metrics. Pure: no I/O, no printing.
    """
    out: list[VariantResult] = []
    h_a, w_a = image_a_u8.shape
    h_b, w_b = image_b_u8.shape

    # Cache preprocessed images so we only compute each one once even though
    # they are used across multiple SIFT configs.
    preproc_a: dict[str, np.ndarray] = {
        v: preprocess(image_a_u8, v) for v in PREPROCESS_VARIANTS
    }
    preproc_b: dict[str, np.ndarray] = {
        v: preprocess(image_b_u8, v) for v in PREPROCESS_VARIANTS
    }

    for preprocess_variant, sift_config in COMBINATIONS:
        a_proc = preproc_a[preprocess_variant]
        b_proc = preproc_b[preprocess_variant]
        sift = make_sift(sift_config)
        pts_a, pts_b, kp_a, kp_b = _match_points(a_proc, b_proc, sift)
        result = VariantResult(
            variant=f"{preprocess_variant}+{sift_config}",
            preprocess_variant=preprocess_variant,
            sift_config=sift_config,
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


# --------------------------------------------------------------------------
# Coverage-first ranking and noise-trap detection
# --------------------------------------------------------------------------
def best_by_coverage(results: Iterable[VariantResult]) -> VariantResult | None:
    """Pick the best combination by coverage-first ranking.

    Ranking key: ``(min(cov_a, cov_b), inlier_count)``, max wins. Only
    successful results are considered; returns ``None`` if none succeeded.
    """
    successes = [r for r in results if r.success]
    if not successes:
        return None
    return max(successes, key=lambda r: (r.min_coverage, r.inlier_count))


def is_noise_trap(result: VariantResult, baseline_kp_count: int) -> bool:
    """Flag a combination as a likely noise trap: many keypoints relative to
    the raw+default baseline, but poor surviving inlier spread.

    "Many keypoints" means ``max(kp_a, kp_b) > _NOISE_TRAP_KP_RATIO * baseline``;
    "poor coverage" means ``min(cov_a, cov_b) < _NOISE_TRAP_MIN_COVERAGE``,
    which a refused combo (coverage 0) also satisfies.
    """
    if baseline_kp_count <= 0:
        return False
    kp = max(result.kp_a, result.kp_b)
    if kp <= _NOISE_TRAP_KP_RATIO * baseline_kp_count:
        return False
    return result.min_coverage < _NOISE_TRAP_MIN_COVERAGE


def _coverage_label(min_cov: int) -> str:
    if min_cov >= _BROAD_MIN_COVERAGE:
        return "broad"
    if min_cov >= 6:
        return "moderate"
    return "clustered"


# --------------------------------------------------------------------------
# Folder discovery, file loading, printing
# --------------------------------------------------------------------------
def _list_dicoms(folder: Path) -> list[Path]:
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
    print("=" * 108)
    print(f"PAIR: {pair.pair_label}")
    print(f"  A: {pair.file_a}  shape={pair.shape_a}")
    print(f"  B: {pair.file_b}  shape={pair.shape_b}")
    print("-" * 108)
    header = (
        f"{'preprocess':<9}  {'sift':<12}  {'kp_a':>5}  {'kp_b':>5}  {'good':>5}  "
        f"{'status':<7}  {'inl':>4}  {'err_px':>7}  "
        f"{'cov_A':>6}  {'cov_B':>6}  {'min_cov':>7}"
    )
    print(header)
    print("-" * len(header))
    baseline_kp = pair.baseline_kp_count()
    for v in pair.variants:
        if v.success:
            status = "OK"
            inl = f"{v.inlier_count:>4d}"
            err = f"{v.mean_epipolar_error:7.3f}"
            cov_a = f"{v.coverage_a:>2d}/16"
            cov_b = f"{v.coverage_b:>2d}/16"
            min_cov = f"{v.min_coverage:>2d}/16"
        else:
            status = "REFUSED"
            inl = "  --"
            err = "    --"
            cov_a = "   --"
            cov_b = "   --"
            min_cov = "    --"
        line = (
            f"{v.preprocess_variant:<9}  {v.sift_config:<12}  "
            f"{v.kp_a:>5d}  {v.kp_b:>5d}  {v.good_matches:>5d}  "
            f"{status:<7}  {inl}  {err}  {cov_a}  {cov_b}  {min_cov}"
        )
        if is_noise_trap(v, baseline_kp):
            line += "   [NOISE TRAP]"
        print(line)
        if not v.success and v.failure_reason:
            print(f"          reason: {v.failure_reason}")
    print()


def _summarize_pair(pair: PairResult) -> str:
    best = best_by_coverage(pair.variants)
    if best is None:
        return (
            f"  {pair.pair_label}: NO combination produced recoverable geometry."
        )
    spread = _coverage_label(best.min_coverage)
    raw_default = next(
        (
            v for v in pair.variants
            if v.preprocess_variant == "raw" and v.sift_config == "sift_default"
        ),
        None,
    )
    if raw_default is not None and raw_default.success:
        delta = best.min_coverage - raw_default.min_coverage
        if best.preprocess_variant == "raw" and best.sift_config == "sift_default":
            note = " (raw+default already best)"
        elif delta > 0:
            note = f" (raw+default also OK but min-cov {raw_default.min_coverage}/16, +{delta})"
        else:
            note = f" (raw+default also OK with same/higher min-cov)"
    else:
        note = " (raw+default FAILED; this combination rescued the pair)"
    return (
        f"  {pair.pair_label}: best {best.variant!r} -> "
        f"min-coverage {best.min_coverage}/16 ({spread}), "
        f"{best.inlier_count} inliers, err {best.mean_epipolar_error:.3f} px{note}"
    )


def _overall_recommendation(pairs: list[PairResult]) -> str:
    if not pairs:
        return "No pairs analyzed."

    lines: list[str] = []
    n = len(pairs)

    # Per-combination success counts
    succ_count: dict[str, int] = {f"{p}+{s}": 0 for p, s in COMBINATIONS}
    broad_count: dict[str, int] = {f"{p}+{s}": 0 for p, s in COMBINATIONS}
    for p in pairs:
        for v in p.variants:
            if v.success:
                succ_count[v.variant] += 1
                if v.min_coverage >= _BROAD_MIN_COVERAGE:
                    broad_count[v.variant] += 1

    lines.append(f"Pairs analyzed: {n}")
    lines.append(
        f"  Coverage threshold for 'broad': min-coverage >= "
        f"{_BROAD_MIN_COVERAGE}/16 across both images."
    )
    lines.append("")
    lines.append(f"  {'combination':<24}  {'OK':>5}  {'broad':>5}")
    for combo in (f"{p}+{s}" for p, s in COMBINATIONS):
        lines.append(
            f"  {combo:<24}  {succ_count[combo]}/{n}  {broad_count[combo]}/{n}"
        )

    # Per-pair best, for the aggregate "did sift_dense help?" comparison.
    paired: list[tuple[VariantResult | None, VariantResult | None]] = []
    for p in pairs:
        default_best = best_by_coverage(
            [v for v in p.variants if v.sift_config == "sift_default"]
        )
        dense_best = best_by_coverage(
            [v for v in p.variants if v.sift_config == "sift_dense"]
        )
        paired.append((default_best, dense_best))

    dense_strictly_better = sum(
        1 for d, e in paired
        if e is not None and (d is None or e.min_coverage > d.min_coverage)
    )
    dense_matches = sum(
        1 for d, e in paired
        if e is not None and d is not None and e.min_coverage == d.min_coverage
    )
    dense_worse = sum(
        1 for d, e in paired
        if d is not None and (e is None or e.min_coverage < d.min_coverage)
    )
    lines.append("")
    lines.append(
        f"  Best-by-coverage per pair, sift_dense vs sift_default:"
    )
    lines.append(
        f"    dense raised min-coverage on {dense_strictly_better}/{n} pair(s)"
    )
    lines.append(f"    same min-coverage on  {dense_matches}/{n} pair(s)")
    lines.append(f"    dense was worse on    {dense_worse}/{n} pair(s)")

    # Noise traps
    traps: list[str] = []
    for p in pairs:
        baseline = p.baseline_kp_count()
        for v in p.variants:
            if is_noise_trap(v, baseline):
                traps.append(
                    f"    {p.pair_label}: {v.variant} -> "
                    f"kp_max {max(v.kp_a, v.kp_b)} vs baseline {baseline}, "
                    f"min-coverage "
                    f"{v.min_coverage if v.success else 0}/16"
                    f"{' (REFUSED)' if not v.success else ''}"
                )
    lines.append("")
    if traps:
        lines.append(f"  NOISE TRAPS ({len(traps)}):")
        lines.extend(traps)
    else:
        lines.append("  No noise traps flagged.")

    # Overall verdict (coverage-first, no spin)
    best_combo = max(
        succ_count.keys(),
        key=lambda c: (broad_count[c], succ_count[c]),
    )
    best_succ = succ_count[best_combo]
    best_broad = broad_count[best_combo]
    all_failed = all(s == 0 for s in succ_count.values())

    lines.append("")
    if all_failed:
        verdict = (
            "Honest conclusion: NO preprocess x SIFT-density combination "
            "recovered usable geometry on any pair. SIFT-based two-view "
            "geometry is insufficient for these images as-is."
        )
    elif best_broad == 0:
        verdict = (
            f"No combination produced BROAD coverage (min-cov >= "
            f"{_BROAD_MIN_COVERAGE}/16) on any pair. Best combination by "
            f"coverage was {best_combo!r} ({best_succ}/{n} pairs OK, none "
            f"broad). Geometry is recoverable but inlier distribution is "
            f"clustered/moderate — not yet trustworthy for clinical use."
        )
    elif dense_strictly_better > dense_worse and dense_strictly_better > 0:
        verdict = (
            f"sift_dense raised min-coverage on more pairs than it hurt "
            f"({dense_strictly_better} better vs {dense_worse} worse). "
            f"Best combination overall: {best_combo!r} ({best_succ}/{n} "
            f"pairs OK, {best_broad}/{n} broad). Worth considering for the "
            f"engine pending a wider real-data sample."
        )
    elif dense_strictly_better == 0 and dense_worse > 0:
        verdict = (
            f"sift_dense did NOT improve min-coverage on any pair and was "
            f"worse on {dense_worse}/{n}. Best combination overall: "
            f"{best_combo!r} ({best_succ}/{n} pairs OK, {best_broad}/{n} "
            f"broad). Denser SIFT added keypoints without converting them "
            f"into broader correspondences."
        )
    else:
        verdict = (
            f"sift_dense neither clearly helped nor hurt min-coverage. "
            f"Best combination overall: {best_combo!r} ({best_succ}/{n} "
            f"pairs OK, {best_broad}/{n} broad)."
        )
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
    print("Parameters:")
    print(f"  CLAHE:        clipLimit={CLAHE_CLIP_LIMIT}, "
          f"tileGridSize={CLAHE_TILE_GRID}")
    print(f"  equalize:     cv2.equalizeHist (global histogram equalization)")
    print(f"  sift_default: cv2.SIFT_create() with OpenCV defaults "
          f"(contrastThreshold=0.04, edgeThreshold=10)")
    print(f"  sift_dense:   cv2.SIFT_create(contrastThreshold="
          f"{SIFT_DENSE_CONTRAST_THRESHOLD}, edgeThreshold="
          f"{SIFT_DENSE_EDGE_THRESHOLD})")
    print()

    pair_results: list[PairResult] = []
    for idx, (pa, pb) in enumerate(_adjacent_pairs(paths), start=1):
        label = f"pair{idx} ({pa.name} + {pb.name})"
        ra = _load_or_error(pa)
        rb = _load_or_error(pb)
        if isinstance(ra, str) or isinstance(rb, str):
            print("=" * 108)
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
        print("PER-PAIR BEST (coverage-first ranking)")
        print("-" * 108)
        for p in pair_results:
            print(_summarize_pair(p))
        print()
        print("OVERALL")
        print("-" * 108)
        print(_overall_recommendation(pair_results))
    print()
    print(
        "NOTE: data for a human to read. No preprocessing or SIFT config has "
        "been baked into the engine."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="characterize_preprocessing",
        description=(
            "Sweep preprocessing (raw/CLAHE/equalize) x SIFT density "
            "(default/dense) on real radiograph pairs and report whether "
            "two-view geometry becomes recoverable with BROAD inlier "
            "coverage. The folder is read locally; no files are copied or "
            "committed."
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
