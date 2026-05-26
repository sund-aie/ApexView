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
  distributions for synthetic extension vs angulation regimes to support
  future threshold selection.
- Pair classifier (`apexview.engine.pair_classifier`): `classify_pair` returns
  a `ClassificationResult` labelling a radiograph pair as `EXTENSION` (with
  the engine's stitched image) or `ANGULATION` (correction not yet
  implemented), keyed on the `StitchResult.inlier_count` from the existing
  stitcher against the tunable `EXTENSION_MIN_INLIERS` threshold (provisional
  value 100, chosen from synthetic characterization).
