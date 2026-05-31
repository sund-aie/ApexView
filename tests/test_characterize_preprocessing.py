"""Tests for the preprocessing + SIFT-density characterization script.

We verify the SCRIPT'S MACHINERY on fully synthetic images: that analyze_pair
covers all 6 preprocess x SIFT-density combinations, that the sift_dense
config is genuinely applied (not a no-op), that the coverage metric stays in
its 0..16 range, that the coverage-first ranking picks the higher
min-coverage, and that the noise-trap flag fires on inflated-keypoints +
low-coverage and stays silent on broad-coverage. We intentionally do NOT
assert "sift_dense beats sift_default" — that is the empirical question for
real radiographs, and asserting it on clean synthetic data would be a
hollow/misleading test.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Load the experiment script as a module (it lives outside the package).
_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "characterize_preprocessing.py"
_spec = importlib.util.spec_from_file_location("characterize_preprocessing", _SCRIPT)
characterize = importlib.util.module_from_spec(_spec)
sys.modules["characterize_preprocessing"] = characterize
assert _spec.loader is not None
_spec.loader.exec_module(characterize)

PREPROCESS_VARIANTS = characterize.PREPROCESS_VARIANTS
SIFT_CONFIGS = characterize.SIFT_CONFIGS
COMBINATIONS = characterize.COMBINATIONS
analyze_pair = characterize.analyze_pair
preprocess = characterize.preprocess
grid_coverage = characterize.grid_coverage
make_sift = characterize.make_sift
best_by_coverage = characterize.best_by_coverage
is_noise_trap = characterize.is_noise_trap
VariantResult = characterize.VariantResult

IMG_H = 480
IMG_W = 640

# Known intrinsics + relative pose for a small multi-depth scene (mirrors
# test_stereo_geometry.py so we get a recoverable pair).
K_A = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
K_B = np.array([[820.0, 0.0, 310.0], [0.0, 820.0, 250.0], [0.0, 0.0, 1.0]])
_THETA = math.radians(8.0)
R_AB = np.array(
    [
        [math.cos(_THETA), 0.0, math.sin(_THETA)],
        [0.0, 1.0, 0.0],
        [-math.sin(_THETA), 0.0, math.cos(_THETA)],
    ]
)
T_AB = np.array([0.6, 0.05, 0.1])


def _texture(seed: int, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.full((h, w), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(150):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(2, 7)), int(rng.integers(0, 256)), -1)
    for _ in range(80):
        x, y = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        s = int(rng.integers(5, 18))
        cv2.rectangle(img, (x, y), (x + s, y + s), int(rng.integers(0, 256)), -1)
    return img


def _plane_homography(depth_z: float) -> np.ndarray:
    return K_B @ (R_AB - np.outer(T_AB, [0.0, 0.0, 1.0]) / depth_z) @ np.linalg.inv(K_A)


def _multidepth_pair(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    depths = [5.0, 8.0, 13.0]
    tex_seeds = [seed + 1, seed + 3, seed + 5]
    img_a = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    img_b = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for depth, tseed in zip(depths, tex_seeds):
        tex = _texture(tseed)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _ in range(25):
            x = int(rng.integers(0, IMG_W - 120))
            y = int(rng.integers(0, IMG_H - 120))
            cv2.rectangle(
                mask,
                (x, y),
                (x + int(rng.integers(60, 160)), y + int(rng.integers(60, 160))),
                255,
                -1,
            )
        Hab = _plane_homography(depth)
        tex_b = cv2.warpPerspective(tex, Hab, (IMG_W, IMG_H))
        mask_b = cv2.warpPerspective(mask, Hab, (IMG_W, IMG_H))
        img_a[mask > 127] = tex[mask > 127]
        img_b[mask_b > 127] = tex_b[mask_b > 127]
    return img_a, img_b


def _degrade(img: np.ndarray, seed: int) -> np.ndarray:
    """Crude low-contrast + noise + blur, vaguely radiograph-like."""
    lowc = (img.astype(np.float32) * 0.35 + 90.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, size=img.shape).astype(np.float32)
    blurred = cv2.GaussianBlur((lowc + noise).clip(0, 255), (3, 3), 0.7)
    return blurred.astype(np.uint8)


def _make_result(
    *,
    preprocess_variant: str = "raw",
    sift_config: str = "sift_default",
    kp_a: int = 500,
    kp_b: int = 500,
    good_matches: int = 100,
    success: bool = True,
    inlier_count: int = 80,
    mean_epipolar_error: float = 0.5,
    coverage_a: int = 10,
    coverage_b: int = 10,
    failure_reason: str = "",
) -> VariantResult:
    return VariantResult(
        variant=f"{preprocess_variant}+{sift_config}",
        preprocess_variant=preprocess_variant,
        sift_config=sift_config,
        kp_a=kp_a,
        kp_b=kp_b,
        good_matches=good_matches,
        success=success,
        failure_reason=failure_reason,
        inlier_count=inlier_count,
        mean_epipolar_error=mean_epipolar_error,
        coverage_a=coverage_a,
        coverage_b=coverage_b,
    )


# --------------------------------------------------------------------------
# Machinery tests
# --------------------------------------------------------------------------
def test_analyze_pair_covers_all_six_combinations():
    a, b = _multidepth_pair()
    results = analyze_pair(a, b)

    assert isinstance(results, list)
    assert len(results) == len(COMBINATIONS) == 6
    pairs_seen = {(r.preprocess_variant, r.sift_config) for r in results}
    assert pairs_seen == set(COMBINATIONS)

    for r in results:
        assert isinstance(r, VariantResult)
        assert r.preprocess_variant in PREPROCESS_VARIANTS
        assert r.sift_config in SIFT_CONFIGS
        assert r.variant == f"{r.preprocess_variant}+{r.sift_config}"
        assert isinstance(r.kp_a, int) and r.kp_a >= 0
        assert isinstance(r.kp_b, int) and r.kp_b >= 0
        assert isinstance(r.good_matches, int) and r.good_matches >= 0
        assert isinstance(r.success, bool)
        if r.success:
            assert isinstance(r.inlier_count, int) and r.inlier_count >= 8
            assert math.isfinite(r.mean_epipolar_error)
            assert 0 <= r.coverage_a <= 16
            assert 0 <= r.coverage_b <= 16
            assert r.min_coverage == min(r.coverage_a, r.coverage_b)
        else:
            assert r.failure_reason


def test_sift_dense_detects_at_least_as_many_keypoints_as_default():
    """Lowering contrastThreshold cannot reduce detections — it only admits
    additional lower-contrast keypoints. This proves sift_dense is actually
    applied and not a no-op alias for the default."""
    img = _degrade(_multidepth_pair()[0], seed=1)
    sift_default = make_sift("sift_default")
    sift_dense = make_sift("sift_dense")
    kp_default = sift_default.detect(img, None)
    kp_dense = sift_dense.detect(img, None)
    assert len(kp_dense) >= len(kp_default), (
        f"sift_dense should detect >= sift_default; "
        f"got dense={len(kp_dense)}, default={len(kp_default)}"
    )
    # And on at least one realistic image they should differ strictly,
    # otherwise the dense config is silently equivalent.
    assert len(kp_dense) > len(kp_default), (
        "sift_dense produced exactly the same keypoint count as sift_default "
        "on a degraded image; the dense config may not be applied"
    )


def test_grid_coverage_returns_value_in_zero_to_sixteen():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, size=(500, 2)).astype(np.float32)
    cov = grid_coverage(pts, h=480, w=640)
    assert isinstance(cov, int)
    assert 0 <= cov <= 16

    corners = np.float32([[0, 0], [639, 0], [0, 479], [639, 479]])
    assert grid_coverage(corners, h=480, w=640) == 4

    empty = np.empty((0, 2), dtype=np.float32)
    assert grid_coverage(empty, h=480, w=640) == 0


def test_preprocess_rejects_bad_variant():
    a, _ = _multidepth_pair()
    try:
        preprocess(a, "not_a_variant")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown variant")


def test_make_sift_rejects_unknown_config():
    try:
        make_sift("sift_supersonic")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown sift config")


def test_analyze_pair_records_failure_when_match_count_below_floor():
    flat_a = np.full((IMG_H, IMG_W), 100, dtype=np.uint8)
    flat_a[0, 0] = 101
    flat_b = np.full((IMG_H, IMG_W), 100, dtype=np.uint8)
    flat_b[1, 1] = 101

    results = analyze_pair(flat_a, flat_b)
    assert len(results) == 6
    assert all(not r.success for r in results)
    assert all(r.failure_reason for r in results)


# --------------------------------------------------------------------------
# Ranking + noise-trap helpers
# --------------------------------------------------------------------------
def test_best_by_coverage_prefers_higher_min_coverage():
    high = _make_result(
        preprocess_variant="clahe",
        sift_config="sift_dense",
        coverage_a=12,
        coverage_b=12,
        inlier_count=40,
    )
    low = _make_result(
        preprocess_variant="raw",
        sift_config="sift_default",
        coverage_a=15,  # cov_a is higher than `high`s
        coverage_b=5,   # but min is 5 — should lose on coverage-first ranking
        inlier_count=200,  # large inlier count must NOT win over min-coverage
    )
    best = best_by_coverage([low, high])
    assert best is high


def test_best_by_coverage_breaks_ties_with_inlier_count():
    a = _make_result(
        preprocess_variant="raw", sift_config="sift_default",
        coverage_a=10, coverage_b=10, inlier_count=80,
    )
    b = _make_result(
        preprocess_variant="clahe", sift_config="sift_dense",
        coverage_a=10, coverage_b=10, inlier_count=120,
    )
    assert best_by_coverage([a, b]) is b


def test_best_by_coverage_returns_none_when_all_failed():
    fails = [
        _make_result(success=False, coverage_a=0, coverage_b=0,
                     failure_reason="x"),
        _make_result(success=False, coverage_a=0, coverage_b=0,
                     failure_reason="y"),
    ]
    assert best_by_coverage(fails) is None


def test_noise_trap_fires_on_high_kp_low_coverage():
    trap = _make_result(
        preprocess_variant="equalize",
        sift_config="sift_dense",
        kp_a=9000,
        kp_b=9000,
        coverage_a=3,
        coverage_b=2,
        inlier_count=10,
    )
    assert is_noise_trap(trap, baseline_kp_count=500) is True


def test_noise_trap_silent_on_broad_coverage_even_if_kp_inflated():
    not_a_trap = _make_result(
        preprocess_variant="clahe",
        sift_config="sift_dense",
        kp_a=9000,
        kp_b=9000,
        coverage_a=12,
        coverage_b=11,
        inlier_count=120,
    )
    assert is_noise_trap(not_a_trap, baseline_kp_count=500) is False


def test_noise_trap_silent_when_kp_not_inflated():
    moderate = _make_result(
        preprocess_variant="raw",
        sift_config="sift_default",
        kp_a=500,
        kp_b=400,
        coverage_a=4,
        coverage_b=3,
        inlier_count=12,
    )
    # max(kp) = 500 == baseline * 1.0, well under the 2x ratio
    assert is_noise_trap(moderate, baseline_kp_count=500) is False
