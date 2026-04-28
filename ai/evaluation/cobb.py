"""Post-hoc Cobb angle derivation from keypoints or segmentation predictions.

Cavekit: cavekit-model-exploration.md R5.

The MaIA ground-truth Cobb angle is defined via the slopes at the upper and
lower inflection points of a fitted spinal midline curve (see
`metrics_*.json::cobb_curve_metrics`). Both derivation functions in this
module follow the same definition:

1. Build a list of per-vertebra centroids (one (x, y) per target vertebra).
2. Fit a low-degree polynomial `x = p(y)` through the centroids.
3. Sample the derivative `dx/dy` densely along the spine.
4. Cobb angle = the difference between the angles of the maximum-slope and
   minimum-slope tangent vectors (each angle measured against the vertical
   axis as `atan(dx/dy)`).

For a straight spine, `dx/dy` is approximately constant and the Cobb angle
collapses to ~0°. For an S-curve, `dx/dy` swings between positive and
negative extrema at the inflection points, recovering the textbook angle.

Inputs:
  * `cobb_from_keypoints(kps)` — `kps` is the (68, 2) array of corners in
    canonical TL/TR/BL/BR order; each vertebra's centroid is the mean of its
    4 corners.
  * `cobb_from_segmentation(mask)` — `mask` is a 2D label map with classes
    `0` background and `1..17` target vertebrae T1..L5; each vertebra's
    centroid is the centroid of its connected pixels.

Both return a scalar in degrees in `[0, 90]`.
"""

from __future__ import annotations

import math

import numpy as np

from ai.preprocessing.keypoints import (
    KEYPOINTS_PER_VERTEBRA,
    TARGET_VERTEBRA_IDS,
    TOTAL_KEYPOINTS,
    _oriented_corners,
)

NUM_TARGET_VERTEBRAE: int = len(TARGET_VERTEBRA_IDS)  # 17

# Polynomial fit hyperparameters. Degree 5 best reproduces MaIA's
# curve-inflection Cobb values across moderate-to-severe scoliosis cases.
# Degrees 3-4 underfit the S-curve inflection zones; degree 6+ oscillates.
# The derivation is not expected to reproduce MaIA's values for mild (<20°)
# cases, because MaIA's algorithm isolates the dominant curve while a
# centroid-based fit measures the full-spine tilt variation.
_POLY_DEGREE: int = 5
_DENSE_SAMPLES: int = 400


def _cobb_from_centroids(centroids_xy: np.ndarray) -> float:
    """Compute Cobb angle from per-vertebra centroids in (x, y) order.

    Args:
        centroids_xy: (N, 2) float array of vertebra centroids in image
            pixel coordinates. Rows containing NaN are dropped.

    Returns:
        Cobb angle in degrees, in [0, 90].
    """
    if centroids_xy.ndim != 2 or centroids_xy.shape[1] != 2:
        raise ValueError(f"centroids must be (N, 2), got {centroids_xy.shape}")

    valid = np.isfinite(centroids_xy).all(axis=1)
    pts = centroids_xy[valid]
    if len(pts) < 5:
        return 0.0

    # Sort top → bottom (smaller y first in image coordinates)
    order = np.argsort(pts[:, 1], kind="mergesort")
    pts = pts[order]
    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)

    # Guard against degenerate y-range
    if (ys.max() - ys.min()) < 1.0:
        return 0.0

    # Deduplicate y values to avoid polyfit SVD singularities on tied ys.
    _, uniq_idx = np.unique(ys, return_index=True)
    if len(uniq_idx) < len(ys):
        pts = pts[np.sort(uniq_idx)]
        xs = pts[:, 0].astype(np.float64)
        ys = pts[:, 1].astype(np.float64)

    deg = min(_POLY_DEGREE, len(pts) - 1)
    coeffs = np.polyfit(ys, xs, deg=deg)         # x as function of y
    dcoeffs = np.polyder(np.poly1d(coeffs))      # dx/dy

    y_dense = np.linspace(ys.min(), ys.max(), _DENSE_SAMPLES)
    slopes = dcoeffs(y_dense)

    s_max = float(slopes.max())
    s_min = float(slopes.min())

    angle_max = math.degrees(math.atan(s_max))
    angle_min = math.degrees(math.atan(s_min))
    cobb = abs(angle_max - angle_min)
    return float(min(cobb, 90.0))


def _keypoints_to_centroids(keypoints: np.ndarray) -> np.ndarray:
    """Collapse 68 corner keypoints into 17 centroid (x, y) points.

    For each vertebra, the centroid is the mean of its 4 corners. If any
    corner is NaN the centroid for that vertebra is NaN.
    """
    if keypoints.shape != (TOTAL_KEYPOINTS, 2):
        raise ValueError(
            f"keypoints must be shape ({TOTAL_KEYPOINTS}, 2), got {keypoints.shape}"
        )
    grouped = keypoints.reshape(NUM_TARGET_VERTEBRAE, KEYPOINTS_PER_VERTEBRA, 2)
    return grouped.mean(axis=1)  # (17, 2)


def cobb_from_keypoints(keypoints: np.ndarray) -> float:
    """Compute Cobb angle from 68 ordered vertebra keypoints.

    Args:
        keypoints: (68, 2) float array, NaN-padded for missing vertebrae.

    Returns:
        Cobb angle in degrees, in [0, 90].
    """
    centroids = _keypoints_to_centroids(keypoints)
    return _cobb_from_centroids(centroids)


def cobb_from_segmentation(
    mask: np.ndarray,
    num_target_classes: int = NUM_TARGET_VERTEBRAE,
) -> float:
    """Compute Cobb angle from a per-pixel multiclass segmentation.

    Args:
        mask: 2D ndarray of integer class labels. 0 = background. Classes
            1..num_target_classes correspond to T1..L5 in order. Higher
            values are ignored.
        num_target_classes: Number of target classes. Default 17.

    Returns:
        Cobb angle in degrees, in [0, 90].
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    centroids = _mask_to_centroids(mask, num_target_classes)
    return _cobb_from_centroids(centroids)


def _cobb_from_centroids_tangent(
    centroids_xy: np.ndarray,
    window: int = 3,
) -> float:
    """Cobb angle via smoothed piecewise tangent differences.

    Instead of fitting a global polynomial, computes local tangent angles
    from consecutive centroid pairs, smooths with a running average, then
    reports the angular range. On GT masks this yields MAE 9.89° vs 15.71°
    for the polynomial method (r=0.831 vs 0.674).

    Args:
        centroids_xy: (N, 2) float array of vertebra centroids.
        window: running-average kernel width for tangent smoothing.

    Returns:
        Cobb angle in degrees, >= 0. Returns 0.0 when fewer than 4
        valid centroids are present.
    """
    if centroids_xy.ndim != 2 or centroids_xy.shape[1] != 2:
        raise ValueError(f"centroids must be (N, 2), got {centroids_xy.shape}")

    valid = np.isfinite(centroids_xy).all(axis=1)
    pts = centroids_xy[valid]
    if len(pts) < 4:
        return 0.0

    order = np.argsort(pts[:, 1], kind="mergesort")
    pts = pts[order].astype(np.float64)

    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])

    k = min(window, len(dx))
    kernel = np.ones(k) / k
    dx_s = np.convolve(dx, kernel, mode="valid")
    dy_s = np.convolve(dy, kernel, mode="valid")

    angles = np.degrees(np.arctan2(dx_s, dy_s))
    return float(angles.max() - angles.min())


def _mask_to_centroids(
    mask: np.ndarray,
    num_target_classes: int = NUM_TARGET_VERTEBRAE,
) -> np.ndarray:
    """Extract per-vertebra centroids from a remapped segmentation mask."""
    centroids = np.full((num_target_classes, 2), np.nan, dtype=np.float64)
    for i in range(num_target_classes):
        cls = i + 1
        ys, xs = np.where(mask == cls)
        if len(ys) == 0:
            continue
        centroids[i, 0] = float(xs.mean())
        centroids[i, 1] = float(ys.mean())
    return centroids


def _raw_mask_to_centroids(
    mask: np.ndarray,
    target_ids: tuple[int, ...] = TARGET_VERTEBRA_IDS,
) -> np.ndarray:
    """Extract per-vertebra centroids from a raw multiclass ID mask."""
    centroids = np.full((len(target_ids), 2), np.nan, dtype=np.float64)
    for i, vid in enumerate(target_ids):
        ys, xs = np.where(mask == vid)
        if len(ys) == 0:
            continue
        centroids[i, 0] = float(xs.mean())
        centroids[i, 1] = float(ys.mean())
    return centroids


def cobb_from_segmentation_tangent(
    mask: np.ndarray,
    num_target_classes: int = NUM_TARGET_VERTEBRAE,
    window: int = 3,
) -> float:
    """Cobb angle from remapped segmentation mask via smoothed tangent method.

    Drop-in replacement for `cobb_from_segmentation` with better accuracy
    on GT masks (MAE 9.89° vs 15.71°, r=0.831 vs 0.674).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    centroids = _mask_to_centroids(mask, num_target_classes)
    return _cobb_from_centroids_tangent(centroids, window=window)


def cobb_from_raw_multiclass_mask_tangent(
    mask: np.ndarray,
    target_ids: tuple[int, ...] = TARGET_VERTEBRA_IDS,
    window: int = 3,
) -> float:
    """Smoothed tangent Cobb directly from a raw multiclass ID mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    centroids = _raw_mask_to_centroids(mask, target_ids)
    return _cobb_from_centroids_tangent(centroids, window=window)


def cobb_from_raw_multiclass_mask(
    mask: np.ndarray,
    target_ids: tuple[int, ...] = TARGET_VERTEBRA_IDS,
) -> float:
    """Convenience: compute Cobb directly from a raw multiclass ID mask.

    Used by the unit tests so they can read the original
    `LabelMultiClass_ID_PNG` files without first remapping them. Numerically
    identical to applying `remap_to_target_classes` then calling
    `cobb_from_segmentation`.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    centroids = _raw_mask_to_centroids(mask, target_ids)
    return _cobb_from_centroids(centroids)


# ──────────────────────────────────────────────────────────────────────────
# Endplate-tilt Cobb derivation (AASCE / BoostNet formulation)
# ──────────────────────────────────────────────────────────────────────────
#
# Reference: Wu et al., "Automatic Landmark Estimation for Adolescent
# Idiopathic Scoliosis Assessment Using BoostNet", MICCAI 2017, and the
# MICCAI AASCE 2019 challenge that standardised the 68-keypoint layout we
# already produce in ai/preprocessing/keypoints.py.
#
# Rationale for existing alongside `_cobb_from_centroids`: the centroid
# polynomial-fit method substitutes "slopes at the inflection points" (the
# MaIA definition) with "global slope extrema of a degree-5 polynomial
# through 17 points". That substitution is exact only on pure single-C
# curves and degrades on double curves, sparse coverage, and near-straight
# spines (documented in notebooks/experiments/audit/audit_findings.ipynb
# §2 — mean |Δ| ≈ 10° on the 120 scoliosis working-set cases).
#
# This function instead computes per-vertebra tilts directly from each
# vertebra's pixel-cloud PCA, then reports the angular range across all
# present vertebrae. On a single dominant curve this matches clinical Cobb
# within a couple of degrees; on balanced double curves it over-estimates
# (the per-segment extension is deferred — see audit notebook §2).
#
# Note: we do NOT reuse `multiclass_mask_to_keypoints` + TL/TR endplate
# labels here because `_oriented_corners` assigns TL/TR by image-space y
# sorting, which fails for near-square vertebrae (where PCA may pick the
# short axis as "major") and for rotations > 45°. The per-vertebra PCA
# below reselects the major eigenvector as "whichever is more vertical",
# which is robust across the full scoliotic tilt range.


def _vertebra_tilt_deg(ys: np.ndarray, xs: np.ndarray) -> float:
    """Return a single vertebra's endplate tilt in degrees.

    Runs PCA on the pixel cloud, picks the eigenvector **closer to
    vertical** as the vertebra's long axis (regardless of eigenvalue —
    this makes the method robust to near-square vertebrae where PCA's
    automatic major/minor assignment is unstable), and returns the
    perpendicular-to-long-axis angle measured against the horizontal
    image axis.

    Args:
        ys: 1D int array of pixel row indices for the vertebra.
        xs: 1D int array of pixel column indices for the vertebra.

    Returns:
        Tilt angle in degrees, in [-90, 90]. Returns NaN if the pixel
        cloud is too small (< 4 pixels) to fit a covariance.
    """
    if len(ys) < 4:
        return float("nan")

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Two candidate axes. np.linalg.eigh returns them in ascending
    # eigenvalue order; ordinarily eigvecs[:, 1] is the major axis.
    # Instead we pick whichever of the two eigenvectors points more
    # vertically — a real vertebra's long axis is always closer to
    # vertical than to horizontal, even under 40°+ scoliotic rotation.
    v0 = eigvecs[:, 0]  # (dx, dy)
    v1 = eigvecs[:, 1]
    vertical_score_0 = abs(v0[1])  # |dy| — larger means more vertical
    vertical_score_1 = abs(v1[1])
    long_axis = v1 if vertical_score_1 >= vertical_score_0 else v0

    # Endplate is perpendicular to the long axis. Its direction vector
    # is (long_axis.dy, -long_axis.dx) — a 90° rotation.
    endplate_dx = long_axis[1]
    endplate_dy = -long_axis[0]

    # Canonicalise endplate_dx > 0 so the tilt sign is consistent across
    # vertebrae (both rotations of a 180°-symmetric endplate vector are
    # equally valid; we pick the one pointing right).
    if endplate_dx < 0:
        endplate_dx, endplate_dy = -endplate_dx, -endplate_dy

    return float(np.degrees(np.arctan2(endplate_dy, endplate_dx)))


def cobb_from_raw_multiclass_mask_endplates(
    mask: np.ndarray,
    target_ids: tuple[int, ...] = TARGET_VERTEBRA_IDS,
) -> float:
    """Endplate-tilt Cobb directly from a raw multiclass ID mask.

    For each target vertebra present in the mask, runs per-pixel PCA to
    find the long axis, computes endplate tilt (perpendicular-to-long-axis
    angle from horizontal), then reports the range of tilts across all
    present vertebrae. This is the single-curve formulation — see module
    docstring for the double-curve caveat.

    Args:
        mask: 2D integer mask carrying raw vertebra IDs.
        target_ids: IDs to measure (default T1..L5 = 6..22).

    Returns:
        Cobb angle in degrees, in [0, 180]. Returns 0.0 when fewer than 2
        vertebrae are present.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    tilts: list[float] = []
    for vid in target_ids:
        ys, xs = np.where(mask == vid)
        if len(ys) == 0:
            continue
        tilt = _vertebra_tilt_deg(ys, xs)
        if np.isfinite(tilt):
            tilts.append(tilt)

    if len(tilts) < 2:
        return 0.0
    arr = np.asarray(tilts, dtype=np.float64)
    return float(arr.max() - arr.min())


def cobb_from_segmentation_endplates(
    mask: np.ndarray,
    num_target_classes: int = NUM_TARGET_VERTEBRAE,
) -> float:
    """Cobb from a remapped segmentation mask via endplate tilts.

    Drop-in replacement for `cobb_from_segmentation` that uses the
    endplate-tilt formulation instead of centroid-polynomial fitting.

    Args:
        mask: 2D integer class mask. 0 = background, 1..num_target_classes
            = T1..L5 in order.
        num_target_classes: Number of target classes. Default 17.
    """
    target_ids = tuple(range(1, num_target_classes + 1))
    return cobb_from_raw_multiclass_mask_endplates(mask, target_ids=target_ids)
