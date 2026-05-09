"""Regression test for Bug A — default ``TARGET_VERTEBRA_IDS`` matches v2 dataset.

The MaIA v1 dataset used raw IDs 6..22 for T1..L5; the v2 dataset uses 1..17.
v2 is the active dataset (CLAUDE.md root §"MaIA Scoliosis Dataset (v2)"); the
v1 default silently zeroed out callers that omit ``target_ids`` on v2 masks.
This test pins the default to the v2 range so it cannot regress.
"""

from __future__ import annotations

import numpy as np

from ai.evaluation.cobb import cobb_from_raw_multiclass_mask
from ai.preprocessing.keypoints import (
    TARGET_VERTEBRA_IDS as KP_TARGET_IDS,
    multiclass_mask_to_keypoints,
)
from ai.preprocessing.segmentation import (
    TARGET_VERTEBRA_IDS as SEG_TARGET_IDS,
    remap_to_target_classes,
)
from ai.visualization import TARGET_VERTEBRA_IDS as VIS_TARGET_IDS

V2_TARGET_IDS = tuple(range(1, 18))


def test_segmentation_default_is_v2() -> None:
    assert SEG_TARGET_IDS == V2_TARGET_IDS


def test_keypoints_default_is_v2() -> None:
    assert KP_TARGET_IDS == V2_TARGET_IDS


def test_visualization_default_is_v2() -> None:
    assert VIS_TARGET_IDS == V2_TARGET_IDS


def test_v2_mask_passes_through_default_remap() -> None:
    """Synthetic v2 mask (IDs 1..17) survives the default remap."""
    mask = np.zeros((50, 30), dtype=np.uint8)
    for i, vid in enumerate(V2_TARGET_IDS):
        mask[i * 2 : i * 2 + 2, :] = vid
    remapped = remap_to_target_classes(mask)
    # Every v2 ID present should land in its 1-based slot — i.e. unchanged.
    np.testing.assert_array_equal(remapped, mask)


def test_v2_mask_yields_finite_keypoints_with_default() -> None:
    """Default keypoint extraction must find vertebrae in a v2 raw mask."""
    mask = np.zeros((100, 60), dtype=np.uint8)
    for i, vid in enumerate(V2_TARGET_IDS):
        mask[i * 5 : i * 5 + 4, 10:50] = vid
    kps = multiclass_mask_to_keypoints(mask)
    finite_per_vertebra = np.isfinite(kps).all(axis=1).reshape(-1, 4).all(axis=1)
    assert int(finite_per_vertebra.sum()) == len(V2_TARGET_IDS)


def test_v2_mask_cobb_finite_with_default() -> None:
    """``cobb_from_raw_multiclass_mask`` must not return NaN on a v2 mask."""
    mask = np.zeros((400, 200), dtype=np.uint8)
    for i, vid in enumerate(V2_TARGET_IDS):
        y = 20 + i * 20
        x_offset = int(15 * np.sin(i / 2.5))
        mask[y : y + 12, 60 + x_offset : 140 + x_offset] = vid
    cobb = cobb_from_raw_multiclass_mask(mask)
    assert np.isfinite(cobb)
    assert cobb > 0.0
