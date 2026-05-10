"""TDD tests for pseudo-label quality filter (Step 1, Path A — pseudo-labeling).

Quality criteria for accepting a pseudo-label from the Phase 1.2 5-fold
ensemble:

    n_distinct_vertebrae_predicted ≥ MIN_VERTEBRAE_FOR_PSEUDO_LABEL  (14)
    mean_foreground_confidence ≥ MIN_MEAN_CONFIDENCE                  (0.70)
    fraction_of_image_predicted_as_vertebra ∈ valid_range              (0.005..0.40)

These thresholds match v2's coverage profile + reject obvious failure
modes (all-background, all-foreground, blurry/uncertain predictions).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.pseudo_label_roboflow import (
    MIN_MEAN_CONFIDENCE,
    MIN_VERTEBRAE_FOR_PSEUDO_LABEL,
    MIN_FG_FRAC,
    MAX_FG_FRAC,
    pseudo_label_passes_quality,
)


def _make_pred(mask: np.ndarray, conf: np.ndarray) -> dict:
    return {"pred_mask": mask, "confidence_map": conf}


def test_accepts_clean_full_spine_prediction() -> None:
    """17 distinct vertebrae, high mean confidence, reasonable FG fraction."""
    h, w = 256, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    # Stack 17 vertebra "bands" (height 5 each), centered horizontally
    for v in range(1, 18):
        y0 = 30 + (v - 1) * 12
        mask[y0:y0 + 5, 200:300] = v
    conf = np.full((h, w), 0.85, dtype=np.float32)
    accepted, reason = pseudo_label_passes_quality(_make_pred(mask, conf))
    assert accepted, reason


def test_rejects_too_few_vertebrae() -> None:
    h, w = 256, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    for v in range(1, 10):   # only 9 vertebrae
        mask[30 + (v - 1) * 12 : 30 + (v - 1) * 12 + 5, 200:300] = v
    conf = np.full((h, w), 0.85, dtype=np.float32)
    accepted, reason = pseudo_label_passes_quality(_make_pred(mask, conf))
    assert not accepted
    assert "vertebrae" in reason.lower()


def test_rejects_low_confidence() -> None:
    h, w = 256, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    for v in range(1, 18):
        mask[30 + (v - 1) * 12 : 30 + (v - 1) * 12 + 5, 200:300] = v
    # Confidence too low
    conf = np.full((h, w), 0.50, dtype=np.float32)
    accepted, reason = pseudo_label_passes_quality(_make_pred(mask, conf))
    assert not accepted
    assert "confidence" in reason.lower()


def test_rejects_all_background() -> None:
    h, w = 256, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    conf = np.full((h, w), 0.99, dtype=np.float32)
    accepted, reason = pseudo_label_passes_quality(_make_pred(mask, conf))
    assert not accepted


def test_rejects_too_much_foreground() -> None:
    h, w = 256, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    # 60% foreground — obvious failure mode
    for v in range(1, 18):
        mask[v * 8 : v * 8 + 7, :] = v
    conf = np.full((h, w), 0.85, dtype=np.float32)
    accepted, reason = pseudo_label_passes_quality(_make_pred(mask, conf))
    assert not accepted
    assert "fg" in reason.lower() or "foreground" in reason.lower()


def test_thresholds_match_documented_values() -> None:
    """Lock the thresholds — anyone changing these must update tests."""
    assert MIN_VERTEBRAE_FOR_PSEUDO_LABEL == 14
    assert MIN_MEAN_CONFIDENCE == 0.70
    assert MIN_FG_FRAC == 0.005
    assert MAX_FG_FRAC == 0.40
