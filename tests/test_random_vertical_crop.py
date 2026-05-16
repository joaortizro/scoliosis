"""Unit tests for ai/preprocessing/transforms.py — vertical crop aug.

Backs the partial-FOV experiment per
[[2026-05-15_partial_fov_experiment_plan]]. Crop semantics:

- Window is sized as ``f * H_bbox`` where ``H_bbox`` is the vertical
  extent of the GT mask foreground.
- Modes: ``top``, ``bottom``, ``mid`` place the window deterministically;
  ``random`` draws ``y_start`` uniformly within the feasible range using
  an injected RNG.
- Image pixels outside the window are zeroed (mimics a partial film).
  Mask is also zeroed outside the window AND vertebrae whose centroid
  falls outside the window are entirely removed (centroid policy in
  plan §"GT policy for partial vertebrae").
- ``f=1.0`` is a strict identity (used as the eval baseline).
"""

from __future__ import annotations

import numpy as np
import torch

from ai.preprocessing.transforms import (
    RandomVerticalCrop,
    deterministic_vertical_crop,
)


def _make_two_vertebra_mask() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.full((1, 100, 50), 0.5)
    mask = torch.zeros(100, 50, dtype=torch.long)
    mask[20:30, 20:40] = 1  # vertebra 1, centroid y=25
    mask[60:70, 20:40] = 2  # vertebra 2, centroid y=65
    return image, mask


def test_f1_is_identity_all_modes() -> None:
    image, mask = _make_two_vertebra_mask()
    for mode in ("top", "bottom", "mid", "random"):
        rng = np.random.default_rng(0)
        out_img, out_mask = deterministic_vertical_crop(
            image, mask, f=1.0, mode=mode, rng=rng,
        )
        assert torch.equal(out_img, image), f"f=1.0 mode={mode} altered image"
        assert torch.equal(out_mask, mask), f"f=1.0 mode={mode} altered mask"


def test_top_mode_window_correct() -> None:
    image, mask = _make_two_vertebra_mask()
    # bbox rows 20..70, H_bbox=50, L=round(0.5*50)=25, window=[20, 45]
    out_img, out_mask = deterministic_vertical_crop(
        image, mask, f=0.5, mode="top",
    )
    assert (out_img[:, :20, :] == 0).all()
    assert (out_img[:, 45:, :] == 0).all()
    assert (out_img[:, 20:45, :] == 0.5).all()
    # vertebra 1 centroid (y=25) inside [20, 45] → kept where mask is within window
    assert (out_mask[20:30, 20:40] == 1).all()
    # vertebra 2 centroid (y=65) outside → entirely removed
    assert int((out_mask == 2).sum()) == 0
    # mask rows outside window zeroed
    assert (out_mask[:20, :] == 0).all()
    assert (out_mask[45:, :] == 0).all()


def test_bottom_mode_window_correct() -> None:
    image, mask = _make_two_vertebra_mask()
    # bbox 20..70, L=25, window=[45, 70]
    out_img, out_mask = deterministic_vertical_crop(
        image, mask, f=0.5, mode="bottom",
    )
    assert (out_img[:, :45, :] == 0).all()
    assert (out_img[:, 70:, :] == 0).all()
    assert (out_img[:, 45:70, :] == 0.5).all()
    # vertebra 1 centroid outside [45, 70] → removed
    assert int((out_mask == 1).sum()) == 0
    # vertebra 2 centroid inside → kept
    assert (out_mask[60:70, 20:40] == 2).all()


def test_mid_mode_centered_on_bbox_midpoint() -> None:
    image = torch.full((1, 100, 50), 0.5)
    mask = torch.zeros(100, 50, dtype=torch.long)
    mask[10:90, 20:40] = 1  # bbox 10..90, midpoint=50, H_bbox=80
    # L=round(0.5*80)=40, window=[30, 70]
    out_img, _ = deterministic_vertical_crop(
        image, mask, f=0.5, mode="mid",
    )
    assert (out_img[:, :30, :] == 0).all()
    assert (out_img[:, 70:, :] == 0).all()
    assert (out_img[:, 30:70, :] == 0.5).all()


def test_random_mode_is_deterministic_with_seeded_rng() -> None:
    image, mask = _make_two_vertebra_mask()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    o1_img, o1_mask = deterministic_vertical_crop(
        image, mask, f=0.4, mode="random", rng=rng1,
    )
    o2_img, o2_mask = deterministic_vertical_crop(
        image, mask, f=0.4, mode="random", rng=rng2,
    )
    assert torch.equal(o1_img, o2_img)
    assert torch.equal(o1_mask, o2_mask)


def test_random_mode_window_inside_bbox() -> None:
    image, mask = _make_two_vertebra_mask()
    # bbox=[20, 70], L=round(0.4*50)=20, window starts in [20, 50]
    for seed in range(10):
        rng = np.random.default_rng(seed)
        out_img, _ = deterministic_vertical_crop(
            image, mask, f=0.4, mode="random", rng=rng,
        )
        # Find non-zero row band
        nonzero_rows = (out_img[0] != 0).any(dim=1).nonzero(as_tuple=True)[0]
        assert nonzero_rows.numel() > 0
        y_start = int(nonzero_rows.min())
        y_end = int(nonzero_rows.max()) + 1
        assert 20 <= y_start
        assert y_end <= 70
        assert (y_end - y_start) == 20


def test_empty_mask_is_passthrough() -> None:
    image = torch.full((1, 100, 50), 0.5)
    mask = torch.zeros(100, 50, dtype=torch.long)
    out_img, out_mask = deterministic_vertical_crop(
        image, mask, f=0.3, mode="top",
    )
    assert torch.equal(out_img, image)
    assert torch.equal(out_mask, mask)


def test_random_vertical_crop_class_shape_invariant() -> None:
    aug = RandomVerticalCrop(
        p=1.0, f_range=(0.5, 1.0), mode="random",
        rng=np.random.default_rng(0),
    )
    image, mask = _make_two_vertebra_mask()
    out_img, out_mask = aug(image, mask)
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape


def test_random_vertical_crop_p0_is_passthrough() -> None:
    aug = RandomVerticalCrop(
        p=0.0, f_range=(0.5, 1.0), mode="random",
        rng=np.random.default_rng(0),
    )
    image, mask = _make_two_vertebra_mask()
    out_img, out_mask = aug(image, mask)
    assert torch.equal(out_img, image)
    assert torch.equal(out_mask, mask)


def test_random_vertical_crop_p1_actually_crops() -> None:
    aug = RandomVerticalCrop(
        p=1.0, f_range=(0.3, 0.5), mode="random",
        rng=np.random.default_rng(0),
    )
    image, mask = _make_two_vertebra_mask()
    out_img, _ = aug(image, mask)
    # At least one image pixel was zeroed (image was 0.5 everywhere; bbox
    # forces a strict subset on f<1)
    assert int((out_img == 0).sum()) > 0


def test_random_vertical_crop_validates_args() -> None:
    import pytest
    with pytest.raises(ValueError):
        RandomVerticalCrop(p=1.5, f_range=(0.5, 1.0), mode="random")
    with pytest.raises(ValueError):
        RandomVerticalCrop(p=0.5, f_range=(0.0, 1.0), mode="random")
    with pytest.raises(ValueError):
        RandomVerticalCrop(p=0.5, f_range=(0.5, 1.5), mode="random")
    with pytest.raises(ValueError):
        RandomVerticalCrop(p=0.5, f_range=(0.5, 1.0), mode="bogus")


def test_f_clamp_to_min_one_pixel() -> None:
    """Very small f on a tiny bbox should still produce a 1-pixel window."""
    image = torch.full((1, 10, 5), 0.5)
    mask = torch.zeros(10, 5, dtype=torch.long)
    mask[4:6, 1:4] = 1  # tiny 2px bbox
    out_img, _ = deterministic_vertical_crop(
        image, mask, f=0.1, mode="top",
    )
    # Window has at least 1 row of original content
    nonzero_rows = (out_img[0] != 0).any(dim=1).nonzero(as_tuple=True)[0]
    assert nonzero_rows.numel() >= 1
