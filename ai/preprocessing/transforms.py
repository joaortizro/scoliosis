"""Vertical-crop transforms for partial-FOV robustness.

Backs the partial-FOV experiment per
[[2026-05-15_partial_fov_experiment_plan]]. Used in two regimes:

- **Training** — ``RandomVerticalCrop`` is a stochastic augmentation
  applied with probability ``p``; the coverage fraction ``f`` is drawn
  uniformly from ``f_range`` each call.
- **Evaluation** — ``deterministic_vertical_crop`` accepts a fixed
  ``(f, mode)`` and an injected ``rng``, so the (f × mode × case) grid
  in ``scripts/eval_partial_fov.py`` is reproducible.

The window length is computed against the spine bounding box of the GT
mask, not the image size, so a 60%-coverage window on a tall spine is
still 60% of the *spine*. Pixels outside the window are zeroed in both
image and mask; vertebrae whose mask centroid falls outside the window
are removed entirely (centroid policy in plan §"GT policy for partial
vertebrae"). ``f = 1.0`` is a strict identity so the eval baseline
matches the existing M0 checkpoint's full-FOV behaviour.
"""

from __future__ import annotations

import numpy as np
import torch

_VALID_MODES = frozenset({"top", "bottom", "mid", "random"})


def _spine_bbox_rows(mask: torch.Tensor) -> tuple[int, int] | None:
    """Vertical extent ``(y_top, y_bot_excl)`` of the spine, or ``None``."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {tuple(mask.shape)}")
    fg = mask > 0
    if not bool(fg.any()):
        return None
    row_has_fg = fg.any(dim=1).nonzero(as_tuple=True)[0]
    return int(row_has_fg.min()), int(row_has_fg.max()) + 1


def _window_for_mode(
    bbox_top: int,
    bbox_bot: int,
    f: float,
    mode: str,
    rng: np.random.Generator | None,
) -> tuple[int, int]:
    h_bbox = bbox_bot - bbox_top
    win_len = max(1, int(round(f * h_bbox)))
    win_len = min(win_len, h_bbox)
    if mode == "top":
        y0 = bbox_top
    elif mode == "bottom":
        y0 = bbox_bot - win_len
    elif mode == "mid":
        mid = (bbox_top + bbox_bot) // 2
        y0 = mid - win_len // 2
    elif mode == "random":
        feasible_start_hi = bbox_bot - win_len
        if feasible_start_hi <= bbox_top:
            y0 = bbox_top
        else:
            generator = rng if rng is not None else np.random.default_rng()
            y0 = int(generator.integers(bbox_top, feasible_start_hi + 1))
    else:
        raise ValueError(f"unknown mode {mode!r}; choices: {sorted(_VALID_MODES)}")
    return y0, y0 + win_len


def deterministic_vertical_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    f: float,
    mode: str = "mid",
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a vertical crop to ``(image, mask)`` per plan semantics.

    Args:
        image: ``(C, H, W)`` float tensor.
        mask: ``(H, W)`` integer label map, background = 0.
        f: coverage fraction in ``(0, 1]``. ``f == 1.0`` returns the
            inputs unchanged.
        mode: one of ``{"top", "bottom", "mid", "random"}``.
        rng: numpy generator used only when ``mode == "random"``.

    Returns:
        ``(out_image, out_mask)`` of the same shapes as the inputs.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; choices: {sorted(_VALID_MODES)}")
    if not (0.0 < f <= 1.0):
        raise ValueError(f"f must be in (0, 1], got {f}")
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got {tuple(image.shape)}")
    if f == 1.0:
        return image, mask
    bbox = _spine_bbox_rows(mask)
    if bbox is None:
        return image, mask
    bbox_top, bbox_bot = bbox
    y_start, y_end = _window_for_mode(bbox_top, bbox_bot, f, mode, rng)

    out_image = torch.zeros_like(image)
    out_image[:, y_start:y_end, :] = image[:, y_start:y_end, :]

    out_mask = torch.zeros_like(mask)
    out_mask[y_start:y_end, :] = mask[y_start:y_end, :]

    # Centroid policy: drop any vertebra whose centroid (within the FULL
    # mask, not the cropped one) falls outside [y_start, y_end].
    labels = torch.unique(mask)
    for label in labels.tolist():
        if label == 0:
            continue
        rows = (mask == label).any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        centroid_y = float(rows.float().mean().item())
        if centroid_y < y_start or centroid_y >= y_end:
            out_mask[out_mask == label] = 0

    return out_image, out_mask


class RandomVerticalCrop:
    """Stochastic vertical crop for training augmentation.

    Each call samples a Bernoulli ``p`` (skip vs apply), then on apply
    samples ``f ~ U(f_min, f_max)`` and delegates to
    ``deterministic_vertical_crop`` with the configured ``mode``.

    Args:
        p: probability of applying the crop per call.
        f_range: ``(f_min, f_max)`` with ``0 < f_min <= f_max <= 1``.
        mode: one of ``{"top", "bottom", "mid", "random"}``; training
            typically uses ``"random"``.
        rng: optional numpy generator; defaults to a per-call ephemeral
            generator (which means training reproducibility relies on
            seeding upstream — same as the rest of ``ai/training/``).
    """

    def __init__(
        self,
        p: float,
        f_range: tuple[float, float],
        mode: str = "random",
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        f_min, f_max = f_range
        if not (0.0 < f_min <= f_max <= 1.0):
            raise ValueError(
                f"f_range must satisfy 0 < f_min <= f_max <= 1, got {f_range}"
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; choices: {sorted(_VALID_MODES)}")
        self.p = float(p)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.mode = mode
        self._rng = rng

    def _gen(self) -> np.random.Generator:
        if self._rng is not None:
            return self._rng
        # Inherit torch's global seed so training reproducibility (set in
        # ``ai.utils.set_seed`` upstream) propagates without a duplicate
        # numpy-side seed contract.
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return np.random.default_rng(seed)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen = self._gen()
        if float(gen.random()) >= self.p:
            return image, mask
        f = float(gen.uniform(self.f_min, self.f_max))
        return deterministic_vertical_crop(image, mask, f=f, mode=self.mode, rng=gen)
