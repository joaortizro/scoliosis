"""Unit tests for ai/preprocessing/roi_crop.py."""

from __future__ import annotations

import numpy as np
import pytest

from ai.preprocessing.roi_crop import (
    crop_image_and_mask,
    roi_from_mask,
    roi_from_yolo,
)


def test_roi_from_mask_basic() -> None:
    mask = np.zeros((100, 50), dtype=np.uint8)
    mask[20:80, 10:40] = 1
    top, bottom, left, right = roi_from_mask(mask, pad_frac=0.0)
    assert (top, bottom, left, right) == (20, 80, 10, 40)


def test_roi_from_mask_pad_clamps() -> None:
    """Padding past the image boundary clamps to the image extent."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # whole image is FG
    top, bottom, left, right = roi_from_mask(mask, pad_frac=0.5)
    assert (top, bottom, left, right) == (0, 10, 0, 10)


def test_roi_from_mask_empty_falls_back_to_full() -> None:
    mask = np.zeros((30, 20), dtype=np.uint8)
    top, bottom, left, right = roi_from_mask(mask)
    assert (top, bottom, left, right) == (0, 30, 0, 20)


def test_crop_image_and_mask_shape_match() -> None:
    image = np.arange(100 * 50, dtype=np.uint8).reshape(100, 50)
    mask = np.zeros((100, 50), dtype=np.uint8)
    mask[20:80, 10:40] = 3
    img_crop, mask_crop, bbox = crop_image_and_mask(image, mask, pad_frac=0.0)
    assert img_crop.shape == mask_crop.shape == (60, 30)
    assert bbox == (20, 80, 10, 40)


def test_roi_from_mask_rejects_3d() -> None:
    with pytest.raises(ValueError):
        roi_from_mask(np.zeros((3, 10, 10), dtype=np.uint8))


def test_roi_from_yolo_gated() -> None:
    with pytest.raises(NotImplementedError):
        roi_from_yolo()
