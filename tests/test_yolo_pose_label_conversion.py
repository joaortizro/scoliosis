"""TDD tests for v2-mask → YOLO-Pose label conversion (Phase 3b spec §5.2).

YOLO-Pose label format (ultralytics):
    class_id cx cy w h kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
All coords normalized to [0, 1]. Visibility: 0=missing, 1=hidden, 2=visible.

For v2: class_id=0 (vertebra-agnostic), 4 keypoints per vertebra (TL, TR, BL, BR).
For Roboflow: dummy keypoints at bbox corners with visibility=0 (kpt loss masked
during pretraining).
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.data_conversion import (
    YoloPoseLabel,
    format_yolo_pose_line,
    multiclass_mask_to_yolo_pose,
    parse_yolo_pose_line,
    roboflow_bbox_line_to_yolo_pose,
)


# -- YoloPoseLabel dataclass --------------------------------------------------


def test_yolo_pose_label_has_required_fields() -> None:
    """A 4-keypoint label has class_id, bbox (cx, cy, w, h), keypoints (4, 3)."""
    lbl = YoloPoseLabel(
        class_id=0,
        cx=0.5,
        cy=0.5,
        w=0.1,
        h=0.05,
        keypoints=np.array(
            [[0.45, 0.475, 2.0], [0.55, 0.475, 2.0], [0.45, 0.525, 2.0], [0.55, 0.525, 2.0]]
        ),
    )
    assert lbl.class_id == 0
    assert lbl.keypoints.shape == (4, 3)


def test_yolo_pose_label_rejects_unnormalized_coords() -> None:
    with pytest.raises(ValueError):
        YoloPoseLabel(
            class_id=0,
            cx=512,  # not normalized
            cy=0.5,
            w=0.1,
            h=0.05,
            keypoints=np.zeros((4, 3)),
        )


# -- multiclass_mask_to_yolo_pose --------------------------------------------


def test_mask_with_one_vertebra_emits_one_label() -> None:
    """Single non-zero class in mask → one YoloPoseLabel with that class's bbox + corners."""
    mask = np.zeros((100, 50), dtype=np.uint8)
    # Place vertebra class 5 as a 10x6 rectangle centered at (50, 25)
    mask[45:55, 22:28] = 5
    labels = multiclass_mask_to_yolo_pose(mask)
    assert len(labels) == 1
    lbl = labels[0]
    assert lbl.class_id == 0  # vertebra-agnostic
    # Bbox is normalized: cx around 25/50=0.5, cy around 50/100=0.5, w around 6/50=0.12, h around 10/100=0.1
    assert lbl.cx == pytest.approx(0.5, abs=0.05)
    assert lbl.cy == pytest.approx(0.5, abs=0.05)
    assert lbl.w == pytest.approx(0.12, abs=0.05)
    assert lbl.h == pytest.approx(0.1, abs=0.05)
    # All 4 keypoints should be visible (v=2) for a fully-present vertebra
    assert (lbl.keypoints[:, 2] == 2).all()


def test_mask_with_multiple_vertebrae_emits_multiple_labels() -> None:
    mask = np.zeros((300, 100), dtype=np.uint8)
    mask[50:60, 40:60] = 1  # T1
    mask[150:160, 40:60] = 5  # T5
    mask[250:260, 40:60] = 17  # L5
    labels = multiclass_mask_to_yolo_pose(mask)
    assert len(labels) == 3
    cys = sorted([lbl.cy for lbl in labels])
    assert cys[0] < cys[1] < cys[2]


def test_mask_skips_classes_outside_target_range() -> None:
    """Classes outside 1..17 (e.g. v1 IDs > 17) are dropped."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[30:40, 30:40] = 22  # outside v2 1..17
    labels = multiclass_mask_to_yolo_pose(mask)
    assert len(labels) == 1


def test_mask_empty_returns_empty_list() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    labels = multiclass_mask_to_yolo_pose(mask)
    assert labels == []


# -- roboflow_bbox_line_to_yolo_pose -----------------------------------------


def test_roboflow_bbox_gets_dummy_keypoints_visibility_zero() -> None:
    """Roboflow has bbox-only labels. Convert to YOLO-Pose with 4 dummy kpts, all v=0."""
    line = "0 0.5 0.5 0.1 0.05"
    lbl = roboflow_bbox_line_to_yolo_pose(line)
    assert lbl.class_id == 0
    assert lbl.cx == 0.5
    assert lbl.cy == 0.5
    assert lbl.w == 0.1
    assert lbl.h == 0.05
    assert lbl.keypoints.shape == (4, 3)
    # All visibility = 0 (ultralytics treats v=0 as "missing", kpt loss is masked)
    assert (lbl.keypoints[:, 2] == 0).all()


def test_roboflow_filter_drops_non_vertebra_classes() -> None:
    """Roboflow has classes 0=Vertebra, 1=scoliosis_spine, 2=normal_spine.
    Only class 0 is used for detection training."""
    vertebra_line = "0 0.5 0.5 0.1 0.05"
    spine_label_line = "1 0.5 0.5 0.8 0.9"
    vlbl = roboflow_bbox_line_to_yolo_pose(vertebra_line)
    assert vlbl.class_id == 0
    # Non-vertebra returns None
    assert roboflow_bbox_line_to_yolo_pose(spine_label_line) is None


# -- format / parse round-trip -----------------------------------------------


def test_yolo_pose_round_trip() -> None:
    """Write a label to YOLO format string, parse it back, ensure equality."""
    original = YoloPoseLabel(
        class_id=0,
        cx=0.5,
        cy=0.5,
        w=0.1,
        h=0.05,
        keypoints=np.array(
            [[0.45, 0.475, 2.0], [0.55, 0.475, 2.0], [0.45, 0.525, 2.0], [0.55, 0.525, 2.0]]
        ),
    )
    line = format_yolo_pose_line(original)
    parsed = parse_yolo_pose_line(line)
    assert parsed.class_id == original.class_id
    assert parsed.cx == pytest.approx(original.cx, abs=1e-5)
    assert parsed.cy == pytest.approx(original.cy, abs=1e-5)
    assert parsed.w == pytest.approx(original.w, abs=1e-5)
    assert parsed.h == pytest.approx(original.h, abs=1e-5)
    np.testing.assert_allclose(parsed.keypoints, original.keypoints, atol=1e-5)


def test_format_yolo_pose_line_emits_17_floats() -> None:
    """class_id + 4 bbox values + 12 kpt values (4 kpts * 3) = 17 tokens."""
    lbl = YoloPoseLabel(
        class_id=0,
        cx=0.5,
        cy=0.5,
        w=0.1,
        h=0.05,
        keypoints=np.zeros((4, 3)),
    )
    line = format_yolo_pose_line(lbl)
    tokens = line.split()
    assert len(tokens) == 17


def test_parse_yolo_pose_rejects_malformed_line() -> None:
    with pytest.raises(ValueError):
        parse_yolo_pose_line("0 0.5 0.5")  # missing fields
