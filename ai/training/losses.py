import torch
import torch.nn.functional as F


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


def seg_loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    bg_weight: float = 0.1,
    ce_weight: float = 0.3,
    dice_weight: float = 0.7,
) -> torch.Tensor:
    class_weights = torch.tensor(
        [bg_weight] + [1.0] * (num_classes - 1),
        device=logits.device,
        dtype=logits.dtype,
    )
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = soft_dice_loss(logits, target, num_classes)
    return ce_weight * ce + dice_weight * dice
