"""Segmentation label remapping.

Cavekit: cavekit-model-exploration.md R4.

The active MaIA v2 multiclass ID PNG masks carry vertebra IDs 1..17 for
T1..L5 (no cervical, no auxiliary entities). The legacy v1 masks used IDs
6..22 — pass ``target_ids=tuple(range(6, 23))`` explicitly when working
with v1 data. ``remap_to_target_classes`` produces an 18-class label map
suitable for training the segmentation model: class 0 = background,
classes 1..17 = T1..L5.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# v2 dataset default: T1..L5 raw IDs are 1..17. For legacy v1 masks pass
# ``target_ids=tuple(range(6, 23))`` explicitly.
TARGET_VERTEBRA_IDS: tuple[int, ...] = tuple(range(1, 18))  # 1..17 inclusive
NUM_SEG_CLASSES: int = 1 + len(TARGET_VERTEBRA_IDS)         # 18 (bg + 17 targets)


def remap_to_target_classes(
    mask: np.ndarray,
    target_ids: Iterable[int] = TARGET_VERTEBRA_IDS,
) -> np.ndarray:
    """Collapse non-target vertebra IDs to background and renumber targets.

    Output classes:
      0           = background
      1..len(ids) = target IDs in given order

    Args:
        mask: 2D integer ndarray with raw multiclass IDs in the source set.
        target_ids: Ordered iterable of source vertebra IDs to keep. Each one
            is mapped to its 1-based position. Defaults to v2 (1..17).

    Returns:
        2D uint8 ndarray of the same shape, values in [0, len(target_ids)].
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    target_ids = tuple(target_ids)
    if len(target_ids) > 255:
        raise ValueError("Too many target IDs for uint8 output")

    out = np.zeros_like(mask, dtype=np.uint8)
    for i, vid in enumerate(target_ids, start=1):
        out[mask == vid] = i
    return out
