"""Detection-first vertebra recognition + Cobb pipeline (Path B).

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md

Submodules:
    data_conversion  — v2 mask / Roboflow bbox → YOLO-Pose labels
    yolo_adapter     — ultralytics YOLOv8-Pose wrapper
    postprocess      — confidence filter, top-N, PCA-axis centroid ordering
"""

from ai.detection.data_conversion import (
    YoloPoseLabel,
    format_yolo_pose_line,
    multiclass_mask_to_yolo_pose,
    parse_yolo_pose_line,
    roboflow_bbox_line_to_yolo_pose,
)

__all__ = [
    "YoloPoseLabel",
    "format_yolo_pose_line",
    "multiclass_mask_to_yolo_pose",
    "parse_yolo_pose_line",
    "roboflow_bbox_line_to_yolo_pose",
]
