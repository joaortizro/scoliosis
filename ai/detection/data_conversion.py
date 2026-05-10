"""v2-mask and Roboflow-bbox label conversion to YOLOv8-Pose format.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §5.2.

YOLO-Pose label format (ultralytics):
    class_id cx cy w h kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...

All bbox + keypoint coords normalized to [0, 1]. Keypoint visibility:
    0 = absent (loss masked)
    1 = present but occluded
    2 = present and visible

For v2 cases we have per-vertebra corners from `multiclass_mask_to_keypoints`,
so visibility = 2. For Roboflow cases we have only bbox supervision, so we
emit 4 dummy keypoints at bbox corners with visibility = 0 — ultralytics
masks the kpt loss for v=0 entries, leaving only bbox loss active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ai.preprocessing.keypoints import (
    KEYPOINTS_PER_VERTEBRA,
    TARGET_VERTEBRA_IDS,
    _oriented_corners,
)

VERTEBRA_CLASS_ID: int = 0       # Roboflow vertebra class
NUM_KPTS_PER_VERTEBRA: int = KEYPOINTS_PER_VERTEBRA  # 4
NUM_TOKENS_PER_LINE: int = 1 + 4 + NUM_KPTS_PER_VERTEBRA * 3  # class + bbox + (x,y,v)*4 = 17


@dataclass
class YoloPoseLabel:
    """A single YOLO-Pose label line: bbox + 4 keypoints.

    All bbox + keypoint coords are normalized to [0, 1].
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    keypoints: np.ndarray   # shape (4, 3): [x_norm, y_norm, visibility]

    def __post_init__(self) -> None:
        for name, val in (("cx", self.cx), ("cy", self.cy), ("w", self.w), ("h", self.h)):
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(
                    f"YoloPoseLabel.{name}={val} not in [0, 1]; "
                    "all coords must be normalized to image dimensions"
                )
        kps = np.asarray(self.keypoints, dtype=np.float64)
        if kps.shape != (NUM_KPTS_PER_VERTEBRA, 3):
            raise ValueError(
                f"keypoints must be shape ({NUM_KPTS_PER_VERTEBRA}, 3), got {kps.shape}"
            )
        # x/y in [0,1] required only when visibility > 0 (loss is masked otherwise)
        visible = kps[:, 2] > 0
        if visible.any():
            xy = kps[visible, :2]
            if (xy < 0.0).any() or (xy > 1.0).any():
                raise ValueError(
                    "visible keypoints must have x/y in [0, 1] (normalized)"
                )
        self.keypoints = kps


def multiclass_mask_to_yolo_pose(mask: np.ndarray) -> list[YoloPoseLabel]:
    """Convert a v2 multiclass mask to a list of YoloPoseLabels.

    For each vertebra ID in TARGET_VERTEBRA_IDS (1..17 for v2 default) that
    is present in the mask:
        - 4 corner keypoints from PCA-oriented bounding box (TL, TR, BL, BR)
        - bbox = tight axis-aligned box around the 4 corners
        - class_id = VERTEBRA_CLASS_ID (vertebra-agnostic)
        - visibility = 2 (visible)

    Args:
        mask: 2D ndarray (H, W) of integer vertebra IDs. Background = 0.

    Returns:
        List of YoloPoseLabel, one per present vertebra. Vertebrae with
        fewer than 4 pixels (insufficient for PCA) are skipped.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    labels: list[YoloPoseLabel] = []
    for vid in TARGET_VERTEBRA_IDS:
        ys, xs = np.where(mask == vid)
        if len(ys) < 4:
            continue
        corners = _oriented_corners(ys, xs)  # (4, 2) in (x, y) image coords
        if not np.isfinite(corners).all():
            continue
        # Bbox = tight axis-aligned box around the 4 corners
        x_min = corners[:, 0].min()
        x_max = corners[:, 0].max()
        y_min = corners[:, 1].min()
        y_max = corners[:, 1].max()
        cx = float((x_min + x_max) / 2 / w)
        cy = float((y_min + y_max) / 2 / h)
        bw = float((x_max - x_min) / w)
        bh = float((y_max - y_min) / h)
        # Normalize keypoints + add visibility = 2
        kpts = np.zeros((NUM_KPTS_PER_VERTEBRA, 3), dtype=np.float64)
        kpts[:, 0] = np.clip(corners[:, 0] / w, 0.0, 1.0)
        kpts[:, 1] = np.clip(corners[:, 1] / h, 0.0, 1.0)
        kpts[:, 2] = 2.0
        labels.append(
            YoloPoseLabel(
                class_id=VERTEBRA_CLASS_ID,
                cx=float(np.clip(cx, 0.0, 1.0)),
                cy=float(np.clip(cy, 0.0, 1.0)),
                w=float(np.clip(bw, 0.0, 1.0)),
                h=float(np.clip(bh, 0.0, 1.0)),
                keypoints=kpts,
            )
        )
    return labels


def roboflow_bbox_line_to_yolo_pose(line: str) -> Optional[YoloPoseLabel]:
    """Convert a Roboflow YOLO bbox line to a YoloPoseLabel with dummy keypoints.

    Roboflow's per-image label files contain three classes:
        0 = Vertebra (per-bbox)
        1 = scoliosis spine (per-image)
        2 = normal spine (per-image)

    Only class 0 (Vertebra) bboxes are useful for detection training.
    Classes 1 and 2 are dropped — they are image-level scoliosis/normal
    flags, not detection annotations.

    The returned label has 4 dummy keypoints at bbox corners with
    visibility = 0 so ultralytics masks the keypoint loss for these
    pretraining labels.

    Args:
        line: One line from a Roboflow .txt label file.

    Returns:
        YoloPoseLabel if line is a vertebra (class 0) annotation;
        None for class 1/2 (spine-level) lines.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 tokens in Roboflow bbox line, got {len(parts)}: {line!r}")
    class_id = int(parts[0])
    if class_id != VERTEBRA_CLASS_ID:
        return None  # drop spine-level labels
    cx, cy, w, h = (float(p) for p in parts[1:5])
    # Dummy keypoints at bbox corners (visibility = 0 → kpt loss masked)
    half_w, half_h = w / 2, h / 2
    kpts = np.array(
        [
            [cx - half_w, cy - half_h, 0.0],  # TL
            [cx + half_w, cy - half_h, 0.0],  # TR
            [cx - half_w, cy + half_h, 0.0],  # BL
            [cx + half_w, cy + half_h, 0.0],  # BR
        ],
        dtype=np.float64,
    )
    kpts[:, :2] = np.clip(kpts[:, :2], 0.0, 1.0)
    return YoloPoseLabel(class_id=class_id, cx=cx, cy=cy, w=w, h=h, keypoints=kpts)


def format_yolo_pose_line(label: YoloPoseLabel) -> str:
    """Serialize a YoloPoseLabel into a single YOLO-Pose label-file line."""
    tokens = [str(label.class_id), f"{label.cx:.6f}", f"{label.cy:.6f}",
              f"{label.w:.6f}", f"{label.h:.6f}"]
    for kpt in label.keypoints:
        tokens.extend([f"{kpt[0]:.6f}", f"{kpt[1]:.6f}", f"{int(kpt[2])}"])
    return " ".join(tokens)


def parse_yolo_pose_line(line: str) -> YoloPoseLabel:
    """Parse a YOLO-Pose label-file line back into a YoloPoseLabel."""
    parts = line.strip().split()
    if len(parts) != NUM_TOKENS_PER_LINE:
        raise ValueError(
            f"expected {NUM_TOKENS_PER_LINE} tokens, got {len(parts)}: {line!r}"
        )
    class_id = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:5])
    kpts = np.zeros((NUM_KPTS_PER_VERTEBRA, 3), dtype=np.float64)
    for i in range(NUM_KPTS_PER_VERTEBRA):
        offset = 5 + i * 3
        kpts[i, 0] = float(parts[offset])
        kpts[i, 1] = float(parts[offset + 1])
        kpts[i, 2] = float(parts[offset + 2])
    return YoloPoseLabel(class_id=class_id, cx=cx, cy=cy, w=w, h=h, keypoints=kpts)
