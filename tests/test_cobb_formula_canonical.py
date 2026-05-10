"""Canonical Cobb-formula tests on synthetic 17-vertebra keypoint sets.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §4 [4] + §7.

Each test constructs a (68, 2) keypoint array (TL, TR, BL, BR per vertebra,
17 vertebrae T1..L5, in canonical order from `multiclass_mask_to_keypoints`),
feeds it to `cobb_from_keypoints_endplate`, and asserts the returned Cobb
matches the analytically-known angle within float tolerance.

Construction: each vertebra has uniform width=W, height=H, centered at
(cx, cy), tilted by midline angle theta_v from vertical. The 4 corners are
the vertices of the rotated rectangle:
    midline direction = (sin(theta), cos(theta))   (theta=0 means vertical)
    perpendicular     = (cos(theta), -sin(theta))
    TL = center - 0.5*H * midline + 0.5*W * perp
    TR = center - 0.5*H * midline - 0.5*W * perp     (note sign convention)
    BL = center + 0.5*H * midline + 0.5*W * perp
    BR = center + 0.5*H * midline - 0.5*W * perp
where +y is downward (image coordinates).

The midline_slope recovered by the implementation should equal theta_v
exactly (modulo floating-point) since the midline goes from
midpoint(TL,TR) to midpoint(BL,BR), which is centered_top to centered_bot,
i.e. exactly along the midline direction.

The canonical Cobb on the slopes-array is:
    Cobb = max(theta) - min(theta) over the 17 vertebrae   (degrees)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai.evaluation.cobb_endplate import (
    compute_midline_slopes,
    cobb_from_keypoints_endplate,
    cobb_from_midline_slopes,
)


VERTEBRA_W = 30.0   # px
VERTEBRA_H = 20.0   # px
VERTEBRA_SPACING_Y = 30.0   # vertical centroid spacing in px (T1 at smallest y)
SPINE_X_BASE = 256.0        # central column


def _build_keypoints(thetas_deg: list[float]) -> np.ndarray:
    """Construct a (68, 2) keypoint array from per-vertebra midline angles.

    Args:
        thetas_deg: length-17 list of midline tilt angles, theta=0 means
            vertical (midline along +y), positive theta rotates the
            midline clockwise (toward +x at bottom). +y is image-down.

    Returns:
        (68, 2) float array. Order: TL_T1, TR_T1, BL_T1, BR_T1, TL_T2, ...
    """
    assert len(thetas_deg) == 17
    out = np.full((68, 2), np.nan, dtype=np.float64)
    cy = 100.0
    for i, theta_deg in enumerate(thetas_deg):
        theta = math.radians(theta_deg)
        cx = SPINE_X_BASE
        # midline direction (top->bot) and perpendicular (left->right at top)
        m = np.array([math.sin(theta), math.cos(theta)])     # (x, y)
        p = np.array([math.cos(theta), -math.sin(theta)])    # (x, y)
        center = np.array([cx, cy])
        # corners
        tl = center - 0.5 * VERTEBRA_H * m + 0.5 * VERTEBRA_W * p
        tr = center - 0.5 * VERTEBRA_H * m - 0.5 * VERTEBRA_W * p
        bl = center + 0.5 * VERTEBRA_H * m + 0.5 * VERTEBRA_W * p
        br = center + 0.5 * VERTEBRA_H * m - 0.5 * VERTEBRA_W * p
        # advisor's TL is "top-left in image" — for theta>0 (clockwise tilt),
        # the geometric TL of the rotated rect IS at center - 0.5H*m + 0.5W*p
        # (top side, left perp). For theta<0 the same definition still holds
        # because keypoints.py canonicalizes to image-left/right post-PCA;
        # we mimic that by re-sorting per row pair.
        # Sort top-pair by x, bottom-pair by x — matches keypoints.py order.
        top = np.array([tl, tr])
        bot = np.array([bl, br])
        top = top[np.argsort(top[:, 0], kind="mergesort")]
        bot = bot[np.argsort(bot[:, 0], kind="mergesort")]
        out[i * 4 : i * 4 + 4] = np.vstack([top[0], top[1], bot[0], bot[1]])
        cy += VERTEBRA_SPACING_Y
    return out


# -- straight spine --------------------------------------------------------


def test_straight_spine_returns_zero_cobb() -> None:
    kps = _build_keypoints([0.0] * 17)
    cobb, _ = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(0.0, abs=1e-6)


def test_uniform_rotation_returns_zero_cobb() -> None:
    """Every vertebra rotated by 30° — no curvature, Cobb = 0."""
    kps = _build_keypoints([30.0] * 17)
    cobb, _ = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(0.0, abs=1e-6)


# -- C-curves (linear tilt progression) -----------------------------------


def test_mild_c_curve_20deg() -> None:
    """Slopes from -10° to +10° linearly across 17 vertebrae → Cobb = 20°."""
    thetas = list(np.linspace(-10.0, 10.0, 17))
    kps = _build_keypoints(thetas)
    cobb, info = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(20.0, abs=0.5)
    # upper end vertebra at chain index 0 (slope = -10), lower end at 16 (slope = +10)
    assert info["upper_end_idx"] == 0
    assert info["lower_end_idx"] == 16


def test_moderate_c_curve_30deg() -> None:
    thetas = list(np.linspace(-15.0, 15.0, 17))
    kps = _build_keypoints(thetas)
    cobb, _ = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(30.0, abs=0.5)


def test_severe_c_curve_50deg() -> None:
    thetas = list(np.linspace(-25.0, 25.0, 17))
    kps = _build_keypoints(thetas)
    cobb, _ = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(50.0, abs=0.5)


# -- S-curve --------------------------------------------------------------


def test_s_curve_dominant_primary_30deg() -> None:
    """S-curve: thoracic (T1..T8) sweeps 0→+15→0; lumbar (T9..L5) sweeps 0→-15→0.

    Naive max(slope)-min(slope) = 15-(-15) = 30°. This IS the dominant
    primary curve in this constructed case (both curves are equal magnitude),
    so canonical and naive agree. We assert the result is 30° to lock the
    formula behavior on a balanced S.
    """
    # 8 thoracic vertebrae: 0, +5, +10, +15, +10, +5, 0, 0  (peak at index 3)
    # 9 lumbar vertebrae:  0, -5, -10, -15, -10, -5, 0, 0, 0  (peak at index 12)
    thetas = [0.0, 5.0, 10.0, 15.0, 10.0, 5.0, 0.0, 0.0,
              -5.0, -10.0, -15.0, -10.0, -5.0, 0.0, 0.0, 0.0, 0.0]
    assert len(thetas) == 17
    kps = _build_keypoints(thetas)
    cobb, info = cobb_from_keypoints_endplate(kps)
    assert cobb == pytest.approx(30.0, abs=0.5)
    # upper end at idx 3 (slope=+15, max), lower end at idx 10 (slope=-15, min)
    assert info["upper_end_idx"] == 3
    assert info["lower_end_idx"] == 10


# -- partial coverage (NaN handling) --------------------------------------


def test_missing_vertebrae_handled_gracefully() -> None:
    """Mid-spine view: only T8..L1 visible (10 vertebrae). Cobb computable
    from valid subset; missing vertebrae must not poison the result."""
    full_thetas = list(np.linspace(-15.0, 15.0, 17))
    kps = _build_keypoints(full_thetas)
    # Mask out T1..T7 and L2..L5 (keep indices 7..12 = T8..L1)
    for i in list(range(0, 7)) + list(range(13, 17)):
        kps[i * 4 : i * 4 + 4] = np.nan
    cobb, info = cobb_from_keypoints_endplate(kps)
    # Visible slopes are at indices 7..12: linspace[-15, 15][7..12] = [-1.875, 1.875] step 3.75
    visible_thetas = full_thetas[7:13]
    expected = max(visible_thetas) - min(visible_thetas)
    assert cobb == pytest.approx(expected, abs=0.5)
    assert info["n_valid_vertebrae"] == 6


# -- compute_midline_slopes unit ------------------------------------------


def test_compute_midline_slopes_matches_input_angles() -> None:
    thetas = [0.0, 5.0, -10.0, 15.0, -20.0] + [0.0] * 12
    kps = _build_keypoints(thetas)
    slopes = compute_midline_slopes(kps)
    assert slopes.shape == (17,)
    np.testing.assert_allclose(slopes, thetas, atol=0.01)


def test_compute_midline_slopes_returns_nan_for_missing() -> None:
    kps = _build_keypoints([10.0] * 17)
    kps[8 * 4 : 8 * 4 + 4] = np.nan
    slopes = compute_midline_slopes(kps)
    assert np.isnan(slopes[8])
    assert np.isfinite(slopes[0])


# -- cobb_from_midline_slopes unit ----------------------------------------


def test_cobb_from_midline_slopes_canonical_pairwise() -> None:
    slopes = np.array([0.0, 5.0, 10.0, 15.0, 10.0, 5.0, 0.0,
                       -5.0, -10.0, -15.0, -10.0, -5.0, 0.0,
                       0.0, 0.0, 0.0, 0.0])
    cobb, info = cobb_from_midline_slopes(slopes)
    assert cobb == pytest.approx(30.0, abs=1e-6)
    assert info["upper_end_idx"] == 3
    assert info["lower_end_idx"] == 9


def test_cobb_from_midline_slopes_minimum_required_vertebrae() -> None:
    """Returns 0.0 if fewer than 4 valid vertebrae (cannot establish a curve)."""
    slopes = np.full(17, np.nan)
    slopes[5] = 10.0
    slopes[10] = -10.0
    cobb, info = cobb_from_midline_slopes(slopes)
    assert cobb == 0.0
    assert info["n_valid_vertebrae"] == 2
