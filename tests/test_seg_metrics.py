"""Unit tests for ai/evaluation/seg_metrics.py."""

from __future__ import annotations

import torch

from ai.evaluation.seg_metrics import (
    confusion_per_class,
    macro_dice_batch_mean,
    macro_dice_per_image,
)


def _logits_from_target(target: torch.Tensor, num_classes: int, certainty: float = 10.0) -> torch.Tensor:
    """Build (B, C, H, W) logits that argmax-decode to target exactly."""
    B, H, W = target.shape
    logits = torch.full((B, num_classes, H, W), -certainty)
    logits.scatter_(1, target.unsqueeze(1), certainty)
    return logits


def test_perfect_prediction_dice_is_one() -> None:
    target = torch.tensor([
        [[0, 1, 1], [0, 1, 1], [2, 2, 0]],
    ], dtype=torch.long)
    logits = _logits_from_target(target, num_classes=3)
    per = macro_dice_per_image(logits, target, num_classes=3)
    assert torch.allclose(per, torch.tensor([1.0]), atol=1e-3)


def test_all_background_returns_zero() -> None:
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    logits = _logits_from_target(target, num_classes=3)
    per = macro_dice_per_image(logits, target, num_classes=3)
    assert per.item() == 0.0  # no foreground signal


def test_disjoint_prediction_dice_is_zero() -> None:
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 0:2, 0:2] = 1  # GT class 1 in one corner
    pred = torch.zeros_like(target)
    pred[0, 2:4, 2:4] = 1    # prediction in opposite corner
    logits = _logits_from_target(pred, num_classes=3)
    per = macro_dice_per_image(logits, target, num_classes=3)
    assert per.item() < 1e-3


def test_only_present_classes_count() -> None:
    """With 17 foreground classes but only class 1 present in GT, the
    per-image Dice equals the class-1 Dice (not a 1/17-weighted version)."""
    H, W = 8, 8
    target = torch.zeros((1, H, W), dtype=torch.long)
    target[0, 0:4, 0:4] = 1
    logits = _logits_from_target(target, num_classes=18)
    per = macro_dice_per_image(logits, target, num_classes=18)
    assert torch.allclose(per, torch.tensor([1.0]), atol=1e-3)


def test_partial_overlap_is_in_range() -> None:
    target = torch.zeros((1, 6, 6), dtype=torch.long)
    target[0, 0:3, 0:3] = 1
    pred = torch.zeros_like(target)
    pred[0, 1:4, 1:4] = 1  # 4-pixel overlap of two 9-pixel regions
    logits = _logits_from_target(pred, num_classes=3)
    per = macro_dice_per_image(logits, target, num_classes=3).item()
    expected = 2 * 4.0 / (9 + 9)
    assert abs(per - expected) < 1e-3


def test_batch_mean_averages_per_image() -> None:
    target = torch.zeros((2, 4, 4), dtype=torch.long)
    target[0, 0:2, 0:2] = 1            # image 0 has FG
    pred = target.clone()
    pred[0, 0:2, 0:2] = 0              # image 0 prediction is wrong
    logits = _logits_from_target(pred, num_classes=3)
    mean = macro_dice_batch_mean(logits, target, num_classes=3).item()
    # image 0: dice=0, image 1: no FG → 0. Mean=0.
    assert mean < 1e-3


def test_confusion_shape_and_consistency() -> None:
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 0:2, 0:2] = 1
    pred = torch.zeros_like(target)
    pred[0, 0:2, 1:3] = 1  # half overlap
    cm = confusion_per_class(pred, target, num_classes=3)
    assert cm["tp"].shape == (3,)
    assert cm["tp"][1].item() == 2  # 2 overlapping pixels
    assert cm["fp"][1].item() == 2
    assert cm["fn"][1].item() == 2
