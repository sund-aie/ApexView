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
