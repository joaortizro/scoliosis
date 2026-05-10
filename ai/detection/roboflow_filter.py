"""Coverage filter for Roboflow pretraining set.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §5.1.

Restrict Roboflow training images to those with ≥ DEFAULT_MIN_VERTEBRAE
detected vertebrae, matching v2's coverage profile. Partial-coverage
Roboflow cases (5-13 vertebrae) are excluded from pretraining but
available for Q-22 OOD evaluation.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MIN_VERTEBRAE: int = 14
VERTEBRA_CLASS_ID: int = 0


def count_vertebrae_in_label_file(label_path: Path) -> int:
    """Count vertebra-class (class_id == 0) annotation lines in a Roboflow label file.

    Roboflow labels are space-separated YOLO bbox format.
    Class 1 (scoliosis spine) and class 2 (normal spine) are ignored —
    those are image-level flags, not vertebra detections.
    """
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    count = 0
    for line in label_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split()[0]
        try:
            if int(first) == VERTEBRA_CLASS_ID:
                count += 1
        except ValueError:
            continue
    return count


def filter_roboflow_split(
    labels_dir: Path,
    min_vertebrae: int = DEFAULT_MIN_VERTEBRAE,
) -> list[str]:
    """Return sorted list of image stems whose label files pass the coverage filter.

    Args:
        labels_dir: Directory containing Roboflow .txt label files
            (typically `data/raw/roboflow_scoliosis_v16/labels/<split>/`).
        min_vertebrae: Minimum vertebra count required to keep the image.

    Returns:
        Sorted list of image stems (label-filename without `.txt`).
    """
    kept: list[str] = []
    for label_path in labels_dir.glob("*.txt"):
        if count_vertebrae_in_label_file(label_path) >= min_vertebrae:
            kept.append(label_path.stem)
    kept.sort()
    return kept
