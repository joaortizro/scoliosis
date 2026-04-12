"""Vertebra keypoint extraction from multiclass ID masks.

Cavekit: cavekit-model-exploration.md R3.

Given a HxW multiclass mask where pixel value == vertebra ID, return a
(17*4, 2) array of keypoints in (x, y) pixel coordinates. The 17 target
vertebrae are IDs 6..22 (T1..L5). For each vertebra, 4 corner keypoints
are emitted in this fixed order:

    [top_left, top_right, bottom_left, bottom_right]

so the full layout is:

    [TL_T1, TR_T1, BL_T1, BR_T1, TL_T2, TR_T2, ..., BR_L5]   # 68 keypoints

If a vertebra is absent from the mask, its 4 keypoints are NaN. The corner
order is canonical (top = smaller y in image coordinates, bottom = larger y).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Target vertebra IDs are fixed by the cavekit and the dataset (T1..L5).
TARGET_VERTEBRA_IDS: tuple[int, ...] = tuple(range(6, 23))  # 6..22 inclusive
KEYPOINTS_PER_VERTEBRA: int = 4
TOTAL_KEYPOINTS: int = len(TARGET_VERTEBRA_IDS) * KEYPOINTS_PER_VERTEBRA  # 68


def _oriented_corners(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Return the 4 corners of the oriented bounding box of a pixel cloud.

    Uses PCA on the (x, y) coordinates to find the long axis of the vertebra,
    then projects pixels onto the major / minor axes to find the extents along
    each axis. The 4 corners are returned in canonical order:

        TL, TR, BL, BR

    where "top" is defined as the side with smaller y after projection back
    into image coordinates.

    Args:
        ys: 1D int array of pixel row indices.
        xs: 1D int array of pixel column indices.

    Returns:
        (4, 2) float array of corners in (x, y) order.
    """
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])  # (N, 2) in (x, y)
    if len(pts) < 4:
        return np.full((4, 2), np.nan, dtype=np.float64)

    center = pts.mean(axis=0)
    centered = pts - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals ascending; major axis is the LAST eigenvector
    minor = eigvecs[:, 0]
    major = eigvecs[:, 1]

    # Project pixels onto major / minor axes
    proj_major = centered @ major
    proj_minor = centered @ minor

    major_min, major_max = float(proj_major.min()), float(proj_major.max())
    minor_min, minor_max = float(proj_minor.min()), float(proj_minor.max())

    # 4 corners in PCA frame
    corners_pca = np.array(
        [
            [minor_min, major_min],  # left,  start of major
            [minor_max, major_min],  # right, start of major
            [minor_min, major_max],  # left,  end   of major
            [minor_max, major_max],  # right, end   of major
        ]
    )

    # Back to image coords
    corners_img = (corners_pca[:, [1]] * major + corners_pca[:, [0]] * minor) + center

    # Sort: top = the two with smaller y, bottom = larger y
    order = np.argsort(corners_img[:, 1], kind="mergesort")
    top_two = corners_img[order[:2]]
    bot_two = corners_img[order[2:]]

    # Within top/bot, left = smaller x
    top_two = top_two[np.argsort(top_two[:, 0], kind="mergesort")]
    bot_two = bot_two[np.argsort(bot_two[:, 0], kind="mergesort")]

    return np.vstack([top_two[0], top_two[1], bot_two[0], bot_two[1]])


def multiclass_mask_to_keypoints(
    mask: np.ndarray,
    target_ids: Iterable[int] = TARGET_VERTEBRA_IDS,
) -> np.ndarray:
    """Convert a 2D multiclass ID mask into 68 ordered keypoints.

    Args:
        mask: 2D ndarray (H, W) of integer vertebra IDs. Background is 0.
        target_ids: Iterable of vertebra IDs to extract corners for. Defaults
            to (6..22) i.e. T1..L5.

    Returns:
        (4 * len(target_ids), 2) float array of keypoints in (x, y) pixel
        coordinates. Missing vertebrae are filled with NaN.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    target_ids = tuple(target_ids)
    out = np.full((len(target_ids) * KEYPOINTS_PER_VERTEBRA, 2), np.nan, dtype=np.float64)

    for i, vid in enumerate(target_ids):
        ys, xs = np.where(mask == vid)
        if len(ys) == 0:
            continue
        corners = _oriented_corners(ys, xs)
        out[i * KEYPOINTS_PER_VERTEBRA : (i + 1) * KEYPOINTS_PER_VERTEBRA] = corners

    return out
