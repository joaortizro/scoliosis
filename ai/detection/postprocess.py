"""Postprocess detected vertebrae into a chain-ordered Cobb angle.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §4 [3-4].

Pipeline:
    detections (N, 4, 2)  ─┐
                           │  → cobb_from_detected_vertebrae
    (per-detection corners,│
     arbitrary order)      │
                           ├→ centroids (N, 2)
                           ├→ order_centroids_by_chain (PCA-axis,
                           │     fallback y-sort if variance < threshold)
                           ├→ midline_slope per detection
                           └→ cobb_from_midline_slopes
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.evaluation.cobb_endplate import (
    MIN_VALID_VERTEBRAE_FOR_COBB,
    cobb_from_midline_slopes,
)

DEFAULT_PCA_VARIANCE_THRESHOLD: float = 0.95


def pca_axis_variance_ratio(centroids: np.ndarray) -> float:
    """Return the fraction of total variance captured by the major axis.

    1.0 = perfectly linear chain; 0.5 = isotropic cloud.
    """
    if centroids.ndim != 2 or centroids.shape[1] != 2 or len(centroids) < 2:
        raise ValueError(f"centroids must be (N>=2, 2), got {centroids.shape}")
    cov = np.cov(centroids.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    total = float(eigvals.sum())
    if total <= 1e-12:
        return 1.0
    return float(eigvals[-1] / total)


def order_centroids_by_chain(
    centroids: np.ndarray,
    pca_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD,
) -> np.ndarray:
    """Return integer indices that reorder centroids head→tail (smaller y first).

    Uses PCA-axis projection for tilted spines; falls back to y-sort when the
    PCA major-axis variance ratio is below `pca_threshold` (near-isotropic
    cloud → axis ambiguous).

    Args:
        centroids: (N, 2) float array of (x, y) centroid positions.
        pca_threshold: Minimum fraction of total variance captured by the
            major axis to use PCA-axis ordering. Below this, fall back
            to y-sort.

    Returns:
        (N,) integer array of indices.
    """
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (N, 2), got {centroids.shape}")
    n = len(centroids)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n < 2:
        return np.array([0], dtype=np.int64)

    ratio = pca_axis_variance_ratio(centroids)
    if ratio >= pca_threshold:
        # Project onto major axis
        center = centroids.mean(axis=0)
        cov = np.cov(centroids.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, -1]   # last column = largest eigenvalue
        # Make sure the major axis points "down" (positive y component) so
        # head→tail matches small-y → large-y
        if major[1] < 0:
            major = -major
        proj = (centroids - center) @ major
        return np.argsort(proj, kind="mergesort").astype(np.int64)
    # Fallback: y-sort
    return np.argsort(centroids[:, 1], kind="mergesort").astype(np.int64)


def _midline_slope_from_4corners(corners: np.ndarray) -> float:
    """Single-vertebra midline slope (degrees) from 4 corners (TL, TR, BL, BR)."""
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return float("nan")
    top_mid = 0.5 * (corners[0] + corners[1])
    bot_mid = 0.5 * (corners[2] + corners[3])
    dx = bot_mid[0] - top_mid[0]
    dy = bot_mid[1] - top_mid[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dx, dy)))


def cobb_from_detected_vertebrae(
    detections: np.ndarray,
    pca_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD,
) -> tuple[float, dict[str, Any]]:
    """End-to-end Cobb computation from detector outputs.

    Args:
        detections: (N, 4, 2) array of corner keypoints per detected vertebra,
            in arbitrary order. Corners follow the canonical TL/TR/BL/BR
            convention (sorted by image y then x within top/bot).
        pca_threshold: PCA variance ratio threshold for ordering.

    Returns:
        (cobb_deg, info). info has:
            - n_valid_vertebrae: number of detections with finite slopes
            - chain_order: indices that reorder `detections` head→tail
            - upper_end_idx / lower_end_idx / apex_idx: chain-position
              indices (in the reordered chain, not the input order) of
              the Cobb end vertebrae and apex
            - max_slope_deg / min_slope_deg
    """
    if detections.ndim != 3 or detections.shape[1:] != (4, 2):
        raise ValueError(f"detections must be (N, 4, 2), got {detections.shape}")
    n = len(detections)
    if n < MIN_VALID_VERTEBRAE_FOR_COBB:
        return 0.0, {
            "n_valid_vertebrae": n,
            "chain_order": np.arange(n, dtype=np.int64),
            "upper_end_idx": -1,
            "lower_end_idx": -1,
            "apex_idx": -1,
            "max_slope_deg": float("nan"),
            "min_slope_deg": float("nan"),
        }

    # Centroids per detection (mean of 4 corners)
    centroids = detections.mean(axis=1)  # (N, 2)
    chain_order = order_centroids_by_chain(centroids, pca_threshold=pca_threshold)

    # Compute midline slopes in chain order
    slopes_chain = np.full(17 if n < 17 else n, np.nan, dtype=np.float64)
    # We re-use cobb_from_midline_slopes which expects shape (17,) — pad with NaN
    chain_len = max(17, n)
    slopes_padded = np.full(chain_len, np.nan, dtype=np.float64)
    for chain_idx, det_idx in enumerate(chain_order):
        slopes_padded[chain_idx] = _midline_slope_from_4corners(detections[det_idx])

    # cobb_from_midline_slopes expects (17,) — if N > 17 we trim, if N < 17 padded already
    if chain_len > 17:
        # Use the first 17 in chain order — for cobb_from_midline_slopes
        # which is internally robust to NaN. But since chain_len > 17 means
        # over-detection, we should use ALL slopes for max-min (more robust).
        # Inline the max-min computation in this case.
        valid = np.isfinite(slopes_padded)
        valid_slopes = slopes_padded[valid]
        if len(valid_slopes) < MIN_VALID_VERTEBRAE_FOR_COBB:
            cobb = 0.0
            info: dict[str, Any] = {
                "n_valid_vertebrae": int(valid.sum()),
                "upper_end_idx": -1,
                "lower_end_idx": -1,
                "apex_idx": -1,
                "max_slope_deg": float("nan"),
                "min_slope_deg": float("nan"),
            }
        else:
            valid_indices = np.where(valid)[0]
            argmax_in_valid = int(np.argmax(valid_slopes))
            argmin_in_valid = int(np.argmin(valid_slopes))
            argmax_idx = int(valid_indices[argmax_in_valid])
            argmin_idx = int(valid_indices[argmin_in_valid])
            cobb = float(valid_slopes[argmax_in_valid] - valid_slopes[argmin_in_valid])
            info = {
                "n_valid_vertebrae": int(valid.sum()),
                "upper_end_idx": min(argmax_idx, argmin_idx),
                "lower_end_idx": max(argmax_idx, argmin_idx),
                "apex_idx": (argmax_idx + argmin_idx) // 2,
                "max_slope_deg": float(valid_slopes[argmax_in_valid]),
                "min_slope_deg": float(valid_slopes[argmin_in_valid]),
            }
        info["chain_order"] = chain_order
        return cobb, info

    cobb, info_inner = cobb_from_midline_slopes(slopes_padded[:17])
    info_inner["chain_order"] = chain_order
    return cobb, info_inner
