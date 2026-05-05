"""Spine-region ROI crop for the segmentation network.

Phase 1.2 of the Dice 0.643 → 0.80 plan. The network sees a tighter
crop around the vertebrae instead of the full chest film, removing
lung air, ribs, and labels that the encoder otherwise has to ignore.

Two paths:

- :func:`roi_from_mask` — derives the bbox from the GT multiclass
  mask. Used at **training time** on MaIA where the GT mask exists.
  No detector dependency.
- :func:`roi_from_yolo` — derives the bbox from a vertebra detector.
  Used at **inference time** when no mask is available. Gated on
  Roboflow asset confirmation; until then this path raises.

Both return ``(top, bottom, left, right)`` pixel indices for use as
``image[top:bottom, left:right]``.
"""

from __future__ import annotations

import numpy as np


def _expand(box: tuple[int, int, int, int], pad_frac: float, h: int, w: int) -> tuple[int, int, int, int]:
    top, bottom, left, right = box
    bh = bottom - top
    bw = right - left
    pad_y = int(round(pad_frac * bh))
    pad_x = int(round(pad_frac * bw))
    top = max(0, top - pad_y)
    bottom = min(h, bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(w, right + pad_x)
    return top, bottom, left, right


def roi_from_mask(
    mask: np.ndarray, pad_frac: float = 0.10
) -> tuple[int, int, int, int]:
    """Bbox of all foreground pixels with ``pad_frac`` padding per side.

    Args:
        mask: 2D label map. Class 0 = background, classes >= 1 = vertebrae.
        pad_frac: padding as a fraction of the bbox dimension (default 10%).

    Returns:
        ``(top, bottom, left, right)`` ready for ``mask[top:bottom, left:right]``.
        Falls back to the full image when no foreground is present.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    h, w = mask.shape
    fg = mask > 0
    if not fg.any():
        return 0, h, 0, w
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    box = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
    return _expand(box, pad_frac=pad_frac, h=h, w=w)


def crop_image_and_mask(
    image: np.ndarray, mask: np.ndarray, pad_frac: float = 0.10
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Apply :func:`roi_from_mask` and return (image_crop, mask_crop, bbox)."""
    top, bottom, left, right = roi_from_mask(mask, pad_frac=pad_frac)
    return image[top:bottom, left:right], mask[top:bottom, left:right], (top, bottom, left, right)


def roi_from_yolo(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Inference-time bbox from a YOLO detector. Gated until Roboflow asset is confirmed."""
    raise NotImplementedError(
        "roi_from_yolo requires the Roboflow scoliosis2.v16i export to train the YOLO "
        "detector. Confirm the asset path with the user before using this path. "
        "Training still works via roi_from_mask (no detector needed)."
    )
