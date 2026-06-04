# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project scaffold: src-layout Python package `apexview` with an empty
  `engine` subpackage, pytest wired up via a `dev` optional dependency group,
  and a single smoke test asserting the package version.
- Extension stitcher (`apexview.engine.extension_stitch`): SIFT + Lowe + RANSAC
  homography recovery for translated same-angle radiograph pairs, returning a
  `StitchResult` with the stitched canvas, homography, inlier count, and mean
  reprojection error (the planar-fit signal a later angulation engine will
  consume).
- Characterization experiment script `experiments/characterize_angulation.py`
  (research tool, not a shipped engine feature) that prints reprojection-error
  AND inlier-count distributions for synthetic extension vs angulation regimes
  to support future threshold selection. The angulation regime renders a
  two-depth 3D scene from a tilted camera so the warp produces genuine
  depth-dependent parallax no single homography can model.
- Pair classifier (`apexview.engine.pair_classifier`): `classify_pair` returns
  a `ClassificationResult` labelling a radiograph pair as `EXTENSION` (with
  the engine's stitched image) or `ANGULATION` (correction not yet
  implemented), keyed on the `StitchResult.inlier_count` from the existing
  stitcher against the tunable `EXTENSION_MIN_INLIERS` threshold (provisional
  value 100, chosen from synthetic characterization).
- Read-only DICOM input adapter (`apexview.io.dicom_reader`): `load_dicom`
  returns a `RadiographImage` carrying the raw pixel array, a per-image
  min/max-rescaled uint8 view ready for the engine, bit depth, and a
  pixel-spacing field that honestly reports `None` when neither
  `ImagerPixelSpacing` nor `PixelSpacing` is present (never fabricates a
  default). Adds `pydicom` as a runtime dependency.
- Command-line interface (`apexview.cli`, installed as the `apexview` console
  script): a thin client over the reader and classifier with an `inspect` mode
  (reports one DICOM's dimensions, bit depth, honest pixel-spacing status, and
  intensity range) and an `analyze` mode (classifies a two-file pair as
  EXTENSION or ANGULATION, optionally saving the engine's stitched PNG via
  `--out`). Prints only engine/reader-owned values, never recomputing them,
  and reports errors as clean messages with exit codes instead of tracebacks.
  Adds `pillow` as a runtime dependency for PNG saving only.
- Two-view stereo geometry (`apexview.engine.stereo_geometry`):
  `estimate_two_view_geometry` recovers the fundamental matrix of an angulated
  pair via the same SIFT + Lowe matching convention as the stitcher, returning
  a `TwoViewGeometry` with F, RANSAC inlier count, mean symmetric epipolar
  error, and matches used. Refuses planar/degenerate scenes via a
  homography-inlier-ratio guard, and pins down the `pts_b^T F pts_a = 0`
  epipolar direction. Foundation math only — no pose, triangulation, or
  correction yet.
- Preprocessing characterization experiment script
  `experiments/characterize_preprocessing.py` (research tool, not a shipped
  feature) that measures whether CLAHE / `equalizeHist` make two-view
  geometry recoverable on real radiograph pairs, reporting Lowe matches,
  RANSAC inlier count, mean epipolar error, and 4x4-grid spatial coverage of
  surviving inliers per variant. The user runs it locally against their own
  DICOM folder; no images enter the repo.
- Extended the preprocessing experiment with a SIFT-density A/B
  (`sift_default` vs `sift_dense` via lowered `contrastThreshold`), sweeping
  6 preprocess x SIFT combinations. Replaced the coverage-blind summary
  with a coverage-first ranking (primary key `min(cov_a, cov_b)`, tiebreaker
  inlier count) and a noise-trap flag for combinations whose keypoints
  inflate without broadening inlier coverage. Still research-only; bakes in
  nothing.
- Shared CLAHE preprocessing (`apexview.engine.preprocessing`): a single
  `preprocess_for_matching` entry point (clipLimit=2.0, tileGridSize=8x8)
  is now applied by default to the SIFT inputs of both `stitch_extension`
  and `estimate_two_view_geometry`. Each exposes `apply_clahe=True` (default)
  so callers can bypass it; the originals are never mutated, and the
  stitched output canvas is built from the caller's original pixels. The
  classifier inherits the default transparently via the stitcher. A denser
  SIFT configuration was evaluated and deliberately NOT adopted as default
  (documented as a constant only, not wired in).
- Relative two-view pose recovery and triangulation
  (`apexview.engine.reconstruction`): `estimate_pose_and_triangulate`
  returns a `RelativeReconstruction` with rotation, unit-norm translation
  direction, a triangulated 3D point cloud in the first camera's frame at
  RELATIVE scale, point count, mean reprojection error, and an
  `intrinsics_were_assumed` flag. Includes an honest `assumed_intrinsics`
  helper for the unknown-K case (focal defaults to max(w, h), principal
  point at center). Builds on A1 via a small additive extension of
  `TwoViewGeometry` to expose inlier correspondences. No metric scale, no
  rectification, no multi-view.
- Detector-plane 2D distance measurement (`apexview.engine.measurement`):
  `measure_distance` converts the pixel distance between two image points
  into millimetres via a `ScaleCalibration`, built either from physical
  sensor size (anisotropic: per-axis mm/px from sensor width/height and the
  image shape) or from a single known reference length (isotropic: assumes
  square pixels). Per-axis conversion is applied before combining, so
  anisotropic pixel spacing is handled correctly. Honest about its limits:
  it is a flat detector-plane measurement, does NOT use the A2
  reconstruction, and is NOT corrected for X-ray magnification or beam
  angulation; every `Measurement` carries `magnification_corrected=False`.
  No metric 3D, no reconstruction dependency, no new runtime dependencies.
