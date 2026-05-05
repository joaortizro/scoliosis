"""Segmentation evaluation metrics.

Macro-Dice over foreground classes 1..N-1 (background dropped, like
:func:`ai.training.losses.soft_dice_loss`). Per-image, per-class so we
can average only over classes that are present in each image — the
v3-corrected dataset has cases with fewer than the full 17 vertebrae,
and including absent classes as zero-Dice would underreport.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ai.preprocessing.segmentation import NUM_SEG_CLASSES


def macro_dice_per_image(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_SEG_CLASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-image macro Dice over present foreground classes.

    Args:
        logits: ``(B, C, H, W)`` raw logits.
        target: ``(B, H, W)`` int64 class labels in ``[0, C-1]``.
        num_classes: number of classes including background.
        eps: smoothing.

    Returns:
        ``(B,)`` tensor — for each image, the mean Dice across its
        present foreground classes. Images with no foreground at all
        return 0.0 (cannot compute).
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B,C,H,W), got {tuple(logits.shape)}")
    if target.dim() != 3:
        raise ValueError(f"target must be (B,H,W), got {tuple(target.shape)}")

    pred = logits.argmax(dim=1)  # (B, H, W)
    pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
    targ_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Drop background channel.
    pred_oh = pred_oh[:, 1:]
    targ_oh = targ_oh[:, 1:]

    dims = (2, 3)
    intersect = (pred_oh * targ_oh).sum(dim=dims)            # (B, C-1)
    cardinality = pred_oh.sum(dim=dims) + targ_oh.sum(dim=dims)
    dice = (2.0 * intersect + eps) / (cardinality + eps)     # (B, C-1)

    # Per-image: average over classes that are present in the GT.
    present = targ_oh.sum(dim=dims) > 0                      # (B, C-1)
    present_count = present.sum(dim=1).clamp(min=1)           # (B,)
    per_image = (dice * present.float()).sum(dim=1) / present_count

    # Images with zero foreground: report 0.0 (no signal).
    has_fg = (targ_oh.sum(dim=(1, 2, 3)) > 0).float()
    return per_image * has_fg


def macro_dice_batch_mean(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_SEG_CLASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean of :func:`macro_dice_per_image` across the batch.

    Returns a 0-d tensor.
    """
    return macro_dice_per_image(logits, target, num_classes=num_classes, eps=eps).mean()


class DatasetDiceAccumulator:
    """Pooled per-class Dice across an entire dataset.

    Matches the metric used in ``model_primer_v3_corrected.ipynb`` so
    trainer numbers are directly comparable to the notebook baseline.
    Pools intersect / cardinality across all batches before computing
    Dice, then averages over classes whose cardinality > 0.
    """

    def __init__(self, num_classes: int = NUM_SEG_CLASSES, device: torch.device | None = None):
        self.num_classes = num_classes
        dev = device if device is not None else torch.device("cpu")
        self.inter = torch.zeros(num_classes - 1, device=dev)
        self.card = torch.zeros(num_classes - 1, device=dev)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        for c in range(1, self.num_classes):
            p = pred == c
            g = target == c
            self.inter[c - 1] += (p & g).sum()
            self.card[c - 1] += p.sum() + g.sum()

    def compute(self) -> float:
        dice_per_cls = (2.0 * self.inter) / self.card.clamp(min=1e-6)
        valid = self.card > 0
        if not valid.any():
            return float("nan")
        return float(dice_per_cls[valid].mean())


def confusion_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_SEG_CLASSES,
) -> dict[str, torch.Tensor]:
    """Per-class TP/FP/FN counts pooled across the batch.

    Args:
        pred: ``(B, H, W)`` int64 predicted labels.
        target: ``(B, H, W)`` int64 GT labels.

    Returns:
        Dict with ``tp``, ``fp``, ``fn`` as ``(num_classes,)`` int64
        tensors. Useful for the diagnostic preview the trainer writes
        per fold.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}")
    pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)
    targ_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)

    dims = (0, 2, 3)
    tp = (pred_oh & targ_oh).sum(dim=dims)
    fp = (pred_oh & ~targ_oh).sum(dim=dims)
    fn = (~pred_oh & targ_oh).sum(dim=dims)
    return {"tp": tp, "fp": fp, "fn": fn}
