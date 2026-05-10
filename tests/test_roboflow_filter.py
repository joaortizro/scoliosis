"""TDD tests for Roboflow coverage filter (Phase 3b spec §5.1).

Restrict Roboflow training images to those with ≥ 14 detected vertebrae,
matching v2's coverage profile. Partial-coverage Roboflow cases (5-13
vertebrae) are excluded from pretraining but available for Q-22 OOD eval.

The filter operates on label files: counts `class_id == 0` (vertebra)
lines per image and returns image stems passing the threshold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai.detection.roboflow_filter import (
    DEFAULT_MIN_VERTEBRAE,
    count_vertebrae_in_label_file,
    filter_roboflow_split,
)


def _write_label(tmp_dir: Path, stem: str, lines: list[str]) -> None:
    (tmp_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


# -- count_vertebrae_in_label_file ------------------------------------------


def test_counts_only_class_zero_vertebra_lines() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.txt"
        p.write_text(
            "0 0.5 0.5 0.1 0.05\n"
            "0 0.4 0.6 0.1 0.05\n"
            "1 0.5 0.5 0.8 0.9\n"   # scoliosis spine — ignore
            "2 0.5 0.5 0.8 0.9\n"   # normal spine — ignore
        )
        assert count_vertebrae_in_label_file(p) == 2


def test_count_handles_empty_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "empty.txt"
        p.write_text("")
        assert count_vertebrae_in_label_file(p) == 0


def test_count_handles_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        count_vertebrae_in_label_file(Path("/nonexistent/file.txt"))


def test_count_ignores_blank_lines_and_whitespace() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ws.txt"
        p.write_text(
            "0 0.5 0.5 0.1 0.05\n"
            "\n"
            "   \n"
            "0 0.4 0.6 0.1 0.05\n"
        )
        assert count_vertebrae_in_label_file(p) == 2


# -- filter_roboflow_split ---------------------------------------------------


def test_filter_split_keeps_only_images_with_min_vertebrae() -> None:
    with tempfile.TemporaryDirectory() as td:
        labels_dir = Path(td) / "labels"
        labels_dir.mkdir()
        _write_label(labels_dir, "img_full", ["0 0.5 0.5 0.1 0.05"] * 17)
        _write_label(labels_dir, "img_partial", ["0 0.5 0.5 0.1 0.05"] * 10)
        _write_label(labels_dir, "img_edge_14", ["0 0.5 0.5 0.1 0.05"] * 14)
        _write_label(labels_dir, "img_edge_13", ["0 0.5 0.5 0.1 0.05"] * 13)
        kept = filter_roboflow_split(labels_dir, min_vertebrae=14)
        kept_set = set(kept)
        assert "img_full" in kept_set
        assert "img_edge_14" in kept_set
        assert "img_partial" not in kept_set
        assert "img_edge_13" not in kept_set
        assert len(kept) == 2


def test_filter_split_returns_sorted_stems() -> None:
    with tempfile.TemporaryDirectory() as td:
        labels_dir = Path(td) / "labels"
        labels_dir.mkdir()
        for s in ["zzz_img", "aaa_img", "mmm_img"]:
            _write_label(labels_dir, s, ["0 0.5 0.5 0.1 0.05"] * 15)
        kept = filter_roboflow_split(labels_dir, min_vertebrae=14)
        assert kept == ["aaa_img", "mmm_img", "zzz_img"]


def test_filter_default_threshold_is_14() -> None:
    assert DEFAULT_MIN_VERTEBRAE == 14
