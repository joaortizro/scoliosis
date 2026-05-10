"""TDD tests for detection postprocessing (Phase 3b spec §4 step [3]).

`order_centroids_by_chain` — sort N detected centroids head→tail using
PCA-axis projection (fallback to y-sort when PCA variance ratio < 0.95).

`cobb_from_detected_vertebrae` — end-to-end: takes (N, 4, 2) detected
corner keypoints (4 corners per vertebra in arbitrary detection order),
returns Cobb angle + chain-ordered indices + apex info.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.postprocess import (
    cobb_from_detected_vertebrae,
    order_centroids_by_chain,
    pca_axis_variance_ratio,
)


# -- pca_axis_variance_ratio -------------------------------------------------


def test_pca_variance_ratio_for_perfectly_vertical_chain() -> None:
    """A chain along the y-axis has PCA variance ratio = 1.0 (all variance on major axis)."""
    centroids = np.column_stack([np.full(10, 100.0), np.linspace(0, 500, 10)])
    ratio = pca_axis_variance_ratio(centroids)
    assert ratio == pytest.approx(1.0, abs=1e-3)


def test_pca_variance_ratio_for_isotropic_cloud() -> None:
    """An isotropic 2D cloud has variance ratio ≈ 0.5 (axes equal)."""
    rng = np.random.default_rng(42)
    centroids = rng.normal(0, 1, (200, 2))
    ratio = pca_axis_variance_ratio(centroids)
    assert 0.4 <= ratio <= 0.6


# -- order_centroids_by_chain ------------------------------------------------


def test_order_recovers_y_sorted_for_vertical_chain() -> None:
    """A perfectly vertical chain shuffles → reorder recovers y-sorted (head=top)."""
    rng = np.random.default_rng(1)
    centroids = np.column_stack([np.full(10, 100.0), np.linspace(0, 500, 10)])
    perm = rng.permutation(10)
    shuffled = centroids[perm]
    order = order_centroids_by_chain(shuffled)
    # Reordering by `order` should recover the original y-sorted chain
    reordered = shuffled[order]
    assert np.all(np.diff(reordered[:, 1]) > 0)


def test_order_handles_tilted_chain() -> None:
    """A diagonal chain (tilted spine) — PCA axis aligns with diagonal."""
    n = 17
    t = np.linspace(0, 1, n)
    centroids = np.column_stack([100 + 50 * t, 100 + 400 * t])  # tilted line
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    order = order_centroids_by_chain(centroids[perm])
    reordered = centroids[perm][order]
    # In the original space, x and y both increase monotonically — reordered
    # should preserve that
    assert np.all(np.diff(reordered[:, 1]) > 0)


def test_order_falls_back_to_ysort_on_low_variance_ratio() -> None:
    """Near-isotropic cloud → PCA axis ambiguous, fallback to y-sort."""
    rng = np.random.default_rng(11)
    cloud = rng.normal(0, 1, (15, 2))
    order = order_centroids_by_chain(cloud, pca_threshold=0.95)
    reordered = cloud[order]
    assert np.all(np.diff(reordered[:, 1]) > 0)


def test_order_returns_indices_not_centroids() -> None:
    centroids = np.array([[10.0, 5.0], [10.0, 1.0], [10.0, 3.0]])
    order = order_centroids_by_chain(centroids)
    assert order.shape == (3,)
    assert order.dtype.kind in {"i", "u"}
    # ordered should give y=1, 3, 5
    assert centroids[order, 1].tolist() == [1.0, 3.0, 5.0]


# -- cobb_from_detected_vertebrae --------------------------------------------


def _build_detections(thetas_deg: list[float]) -> np.ndarray:
    """Construct (N, 4, 2) corner keypoints from per-vertebra midline angles.

    Mirrors the construction in test_cobb_formula_canonical, but returns in
    detection-style (N, 4, 2) without padding to 68.
    """
    import math

    n = len(thetas_deg)
    out = np.zeros((n, 4, 2), dtype=np.float64)
    cy = 100.0
    cx = 256.0
    W, H = 30.0, 20.0
    for i, theta_deg in enumerate(thetas_deg):
        theta = math.radians(theta_deg)
        m = np.array([math.sin(theta), math.cos(theta)])
        p = np.array([math.cos(theta), -math.sin(theta)])
        center = np.array([cx, cy])
        tl = center - 0.5 * H * m + 0.5 * W * p
        tr = center - 0.5 * H * m - 0.5 * W * p
        bl = center + 0.5 * H * m + 0.5 * W * p
        br = center + 0.5 * H * m - 0.5 * W * p
        top = np.array([tl, tr])
        bot = np.array([bl, br])
        top = top[np.argsort(top[:, 0], kind="mergesort")]
        bot = bot[np.argsort(bot[:, 0], kind="mergesort")]
        out[i] = np.vstack([top[0], top[1], bot[0], bot[1]])
        cy += 30.0
    return out


def test_cobb_from_detected_vertebrae_straight_returns_zero() -> None:
    detections = _build_detections([0.0] * 17)
    cobb, info = cobb_from_detected_vertebrae(detections)
    assert cobb == pytest.approx(0.0, abs=1e-5)


def test_cobb_from_detected_vertebrae_mild_c_curve() -> None:
    detections = _build_detections(list(np.linspace(-10.0, 10.0, 17)))
    cobb, info = cobb_from_detected_vertebrae(detections)
    assert cobb == pytest.approx(20.0, abs=0.5)


def test_cobb_from_detected_vertebrae_handles_shuffled_input() -> None:
    """Detector emits vertebrae in arbitrary confidence order — postprocess sorts them."""
    rng = np.random.default_rng(3)
    detections = _build_detections(list(np.linspace(-15.0, 15.0, 17)))
    perm = rng.permutation(17)
    cobb, info = cobb_from_detected_vertebrae(detections[perm])
    assert cobb == pytest.approx(30.0, abs=0.5)


def test_cobb_from_detected_vertebrae_with_fewer_than_min_returns_zero() -> None:
    detections = _build_detections([0.0, 5.0, -5.0])  # only 3 vertebrae
    cobb, info = cobb_from_detected_vertebrae(detections)
    assert cobb == 0.0
    assert info["n_valid_vertebrae"] == 3


def test_cobb_from_detected_vertebrae_returns_chain_indices() -> None:
    detections = _build_detections(list(np.linspace(-15.0, 15.0, 17)))
    rng = np.random.default_rng(9)
    perm = rng.permutation(17)
    cobb, info = cobb_from_detected_vertebrae(detections[perm])
    # info should contain the head→tail chain order so callers can
    # introspect which detection became which chain position
    assert "chain_order" in info
    assert len(info["chain_order"]) == 17
    # The chain order, when applied to detections[perm], should reconstruct
    # the original (centroids monotone in y)
    reordered = detections[perm][info["chain_order"]]
    centroids_y = reordered[:, :, 1].mean(axis=1)
    assert np.all(np.diff(centroids_y) > 0)
