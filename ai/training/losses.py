"""Segmentation losses.

Phase 0 stack:
- ``soft_dice_loss`` — multiclass soft Dice over foreground.
- ``boundary_loss_sdhl`` — Signed Distance Hausdorff-style boundary loss
  (cheap CPU implementation: signed-distance transform of the GT one-hot
  weights the predicted softmax).
- ``seg_loss_fn`` — composite: ce_weight*CE + dice_weight*Dice + boundary_lambda*Boundary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, target.unsqueeze(1), 1.0)
    probs, onehot = probs[:, 1:], onehot[:, 1:]
    dims = (0, 2, 3)
    intersect = (probs * onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2.0 * intersect + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def _signed_distance_map(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class signed distance transform of the GT one-hot.

    Inside the class: +d (negated below). Outside: -d. Background
    channel is dropped, matching :func:`soft_dice_loss`.

    Returns ``(B, C-1, H, W)`` float tensor on the same device as
    ``target``.
    """
    B, H, W = target.shape
    out = np.zeros((B, num_classes - 1, H, W), dtype=np.float32)
    target_np = target.detach().cpu().numpy()
    for b in range(B):
        for c in range(1, num_classes):
            mask = (target_np[b] == c)
            if not mask.any():
                continue  # all-zero distance is fine
            inside = distance_transform_edt(mask)
            outside = distance_transform_edt(~mask)
            out[b, c - 1] = outside - inside
    return torch.from_numpy(out).to(target.device)


def boundary_loss_sdhl(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Boundary loss using GT-side signed distance maps.

    L_b = mean over (B, C-1, H, W) of  softmax_pred * sdt(target)

    The signed distance map is positive *outside* the class region, so
    reducing it pushes the prediction's mass toward the GT boundary.
    Reference: Kervadec et al. "Boundary loss for highly unbalanced
    segmentation" (2019); SDHL extension applied per-class.
    """
    probs = F.softmax(logits, dim=1)[:, 1:]  # drop bg
    sdt = _signed_distance_map(target, num_classes)
    return (probs * sdt).mean()


def seg_loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    bg_weight: float = 0.1,
    ce_weight: float = 0.3,
    dice_weight: float = 0.7,
    boundary_lambda: float = 0.0,
) -> torch.Tensor:
    class_weights = torch.tensor(
        [bg_weight] + [1.0] * (num_classes - 1),
        device=logits.device,
        dtype=logits.dtype,
    )
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = soft_dice_loss(logits, target, num_classes)
    loss = ce_weight * ce + dice_weight * dice
    if boundary_lambda > 0.0:
        loss = loss + boundary_lambda * boundary_loss_sdhl(logits, target, num_classes)
    return loss
