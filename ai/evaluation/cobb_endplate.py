"""Endplate-slope Cobb angle computation — canonical + robust variants.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §4 step [4].

Replaces the centroid-polynomial + tangent pipeline (`cobb_from_keypoints` in
`cobb.py`) with a per-vertebra midline-slope formulation that matches the
literature SOTA for detection-first Cobb measurement.

Algorithm:
    1. For each of the 17 target vertebrae (T1..L5), compute the midline
       direction from top-edge midpoint (mid of TL, TR) to bottom-edge
       midpoint (mid of BL, BR).
    2. midline_slope_deg = degrees(atan2(midline.x, midline.y)) where +y is
       image-down. theta = 0 means vertical midline; theta > 0 means the
       midline tilts toward +x at the bottom (clockwise lean).
    3. Cobb angle = max(slope) - min(slope) across the valid (finite-slope)
       vertebrae in chain order. This is equivalent to canonical pairwise
       max_{i ≤ j} |slope[i] - slope[j]| since |a - b| is symmetric.
    4. Upper end vertebra = whichever of (argmax_slope, argmin_slope) has
       the smaller chain index. Lower end vertebra = the other.
    5. Apex vertebra = argmax of |d²slope / dv²| within
       [upper_end_idx, lower_end_idx].

Returns 0.0 when fewer than 4 valid vertebrae are available — matches the
established convention in `cobb.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.preprocessing.keypoints import (
    KEYPOINTS_PER_VERTEBRA,
    TARGET_VERTEBRA_IDS,
)

NUM_TARGET_VERTEBRAE: int = len(TARGET_VERTEBRA_IDS)  # 17
MIN_VALID_VERTEBRAE_FOR_COBB: int = 4


def compute_midline_slopes(keypoints: np.ndarray) -> np.ndarray:
    """Return per-vertebra midline slopes in degrees.

    Args:
        keypoints: (68, 2) float array of corner keypoints in
            [TL, TR, BL, BR] order, 17 vertebrae × 4 corners. Missing
            vertebrae rows are NaN.

    Returns:
        (17,) float array of midline slopes in degrees. theta = 0 means
        vertical (midline along +y in image coords); theta > 0 means the
        midline tilts toward +x at the bottom. NaN where the vertebra is
        missing or the midline is degenerate.
    """
    if keypoints.shape != (NUM_TARGET_VERTEBRAE * KEYPOINTS_PER_VERTEBRA, 2):
        raise ValueError(
            f"keypoints must be ({NUM_TARGET_VERTEBRAE * KEYPOINTS_PER_VERTEBRA}, 2), "
            f"got {keypoints.shape}"
        )

    slopes = np.full(NUM_TARGET_VERTEBRAE, np.nan, dtype=np.float64)
    for v in range(NUM_TARGET_VERTEBRAE):
        block = keypoints[v * KEYPOINTS_PER_VERTEBRA : (v + 1) * KEYPOINTS_PER_VERTEBRA]
        if not np.isfinite(block).all():
            continue
        tl, tr, bl, br = block[0], block[1], block[2], block[3]
        top_mid = 0.5 * (tl + tr)
        bot_mid = 0.5 * (bl + br)
        dx = bot_mid[0] - top_mid[0]
        dy = bot_mid[1] - top_mid[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        slopes[v] = float(np.degrees(np.arctan2(dx, dy)))
    return slopes


def cobb_from_midline_slopes(slopes: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Compute Cobb from per-vertebra midline slopes (canonical pairwise-max).

    Args:
        slopes: (17,) float array of midline slopes in degrees, NaN where
            vertebra is missing.

    Returns:
        (cobb_deg, info) where info has keys:
            - upper_end_idx: chain index of the more-headward end vertebra
            - lower_end_idx: chain index of the more-tailward end vertebra
            - apex_idx: chain index of the apex vertebra (max curvature
              between the two end vertebrae)
            - n_valid_vertebrae: number of finite slopes
            - max_slope_deg: argmax slope value (NaN if insufficient data)
            - min_slope_deg: argmin slope value (NaN if insufficient data)

        Returns (0.0, {...}) when fewer than MIN_VALID_VERTEBRAE_FOR_COBB
        vertebrae have finite slopes.
    """
    if slopes.shape != (NUM_TARGET_VERTEBRAE,):
        raise ValueError(f"slopes must be ({NUM_TARGET_VERTEBRAE},), got {slopes.shape}")

    valid_mask = np.isfinite(slopes)
    n_valid = int(valid_mask.sum())
    info: dict[str, Any] = {
        "n_valid_vertebrae": n_valid,
        "upper_end_idx": -1,
        "lower_end_idx": -1,
        "apex_idx": -1,
        "max_slope_deg": float("nan"),
        "min_slope_deg": float("nan"),
    }
    if n_valid < MIN_VALID_VERTEBRAE_FOR_COBB:
        return 0.0, info

    valid_indices = np.where(valid_mask)[0]
    valid_slopes = slopes[valid_mask]

    argmax_in_valid = int(np.argmax(valid_slopes))
    argmin_in_valid = int(np.argmin(valid_slopes))
    argmax_idx = int(valid_indices[argmax_in_valid])
    argmin_idx = int(valid_indices[argmin_in_valid])
    max_slope = float(valid_slopes[argmax_in_valid])
    min_slope = float(valid_slopes[argmin_in_valid])

    cobb = max_slope - min_slope
    upper_end_idx = min(argmax_idx, argmin_idx)
    lower_end_idx = max(argmax_idx, argmin_idx)

    # Apex = argmax |d²slope/dv²| in the chain segment [upper_end_idx, lower_end_idx]
    if lower_end_idx - upper_end_idx >= 2:
        # Use the valid slopes within the segment
        seg_mask = valid_mask & (np.arange(NUM_TARGET_VERTEBRAE) >= upper_end_idx) & (
            np.arange(NUM_TARGET_VERTEBRAE) <= lower_end_idx
        )
        seg_indices = np.where(seg_mask)[0]
        if len(seg_indices) >= 3:
            seg_slopes = slopes[seg_indices]
            d2 = np.gradient(np.gradient(seg_slopes))
            apex_idx_in_seg = int(np.argmax(np.abs(d2)))
            apex_idx = int(seg_indices[apex_idx_in_seg])
        else:
            apex_idx = (upper_end_idx + lower_end_idx) // 2
    else:
        apex_idx = (upper_end_idx + lower_end_idx) // 2

    info["upper_end_idx"] = upper_end_idx
    info["lower_end_idx"] = lower_end_idx
    info["apex_idx"] = apex_idx
    info["max_slope_deg"] = max_slope
    info["min_slope_deg"] = min_slope
    return float(cobb), info


def cobb_segment_aware(slopes: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Segment-aware Cobb — handles S-curves correctly.

    Detects inflection points (local extrema of the smoothed slope sequence)
    and computes the largest |slope_max - slope_min| within any single
    monotonic segment. Avoids the pairwise-max inflation on S-curves where
    `max(slope) - min(slope)` would combine two opposite curves.

    Algorithm:
        1. Drop NaN, optionally smooth with median filter (k=3)
        2. Walk the sequence, splitting at sign changes of consecutive
           slope differences (these are slope-direction reversals = the
           classical inflection points of the spinal curve)
        3. For each segment between reversals (inclusive of endpoints),
           compute max(slope) - min(slope)
        4. Return the largest such value
    """
    valid = np.isfinite(slopes)
    n_valid = int(valid.sum())
    info: dict[str, Any] = {
        "n_valid_vertebrae": n_valid,
        "n_segments": 0,
        "primary_segment_range_deg": float("nan"),
    }
    if n_valid < MIN_VALID_VERTEBRAE_FOR_COBB:
        return 0.0, info

    s = slopes[valid].astype(np.float64)

    # Count slope-direction reversals to distinguish C-curve from S-curve
    diffs = np.diff(s)
    reversals = 0
    prev_sign = 0
    for d in diffs:
        sign = int(np.sign(d))
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            reversals += 1
        prev_sign = sign

    # C-curve heuristic: ≤ 1 reversal → treat as single curve, full max-min
    if reversals <= 1:
        primary = float(s.max() - s.min())
        info["n_segments"] = 1
        info["primary_segment_range_deg"] = primary
        return primary, info

    # S-curve / multi-curve: split at zero-crossings of slope
    # (where the spine transitions from one curve direction to another)
    crossing_idx: list[int] = [0]
    for i in range(1, len(s)):
        if (s[i - 1] > 0 and s[i] < 0) or (s[i - 1] < 0 and s[i] > 0) or s[i] == 0:
            crossing_idx.append(i)
    crossing_idx.append(len(s) - 1)
    crossing_idx = sorted(set(crossing_idx))

    segment_ranges: list[float] = []
    for k in range(len(crossing_idx) - 1):
        a, b = crossing_idx[k], crossing_idx[k + 1]
        seg = s[a : b + 1]
        if len(seg) < 2:
            continue
        segment_ranges.append(float(seg.max() - seg.min()))

    if not segment_ranges:
        # Fall back to full max-min
        primary = float(s.max() - s.min())
        info["n_segments"] = 1
        info["primary_segment_range_deg"] = primary
        return primary, info

    primary = max(segment_ranges)
    info["n_segments"] = len(segment_ranges)
    info["primary_segment_range_deg"] = primary
    return primary, info


def cobb_pctile_trim(
    slopes: np.ndarray,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> tuple[float, dict[str, Any]]:
    """Percentile-trimmed pairwise-max Cobb.

    Drops slopes below `lo_pct` and above `hi_pct` percentiles before
    computing max-min. Suppresses single-vertebra outliers from
    PCA-flipped corner regression without changing the multi-curve
    geometry inflation issue (use `cobb_segment_aware` for that).
    """
    valid = np.isfinite(slopes)
    info: dict[str, Any] = {
        "n_valid_vertebrae": int(valid.sum()),
        "lo_pct": lo_pct,
        "hi_pct": hi_pct,
    }
    if valid.sum() < MIN_VALID_VERTEBRAE_FOR_COBB:
        return 0.0, info
    s = slopes[valid]
    lo = np.percentile(s, lo_pct)
    hi = np.percentile(s, hi_pct)
    return float(hi - lo), info


def cobb_from_keypoints_endplate(keypoints: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Convenience: keypoints → midline slopes → Cobb.

    Args:
        keypoints: (68, 2) float array, see `compute_midline_slopes`.

    Returns:
        (cobb_deg, info) — see `cobb_from_midline_slopes`.
    """
    slopes = compute_midline_slopes(keypoints)
    return cobb_from_midline_slopes(slopes)
