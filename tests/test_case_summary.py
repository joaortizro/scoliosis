"""Unit tests for ``ai.visualization.render_case_summary``.

Cavekit: cavekit-case-visualization.md R5 / build-site task T-108.

Three R5 acceptance criteria are covered here:

- **(a) ok path** — synthetic 256×128 image, multiclass mask containing all
  17 target vertebra IDs (6..22), and a fake 2-column curve. The render
  must return ``render_status == "ok"``, exactly six axes, and emit no
  warnings during the call.
- **(b) empty-mask partial** — same image, an all-zero mask. The render
  must return ``render_status == "partial"``, ``render_notes`` must
  mention the empty mask, and the mask subplot must contain a placeholder
  text element.
- **(c) missing-curve partial** — Normal-class synthetic case, non-empty
  mask, ``curve=None``. The render must return ``render_status == "partial"``,
  ``render_notes`` must mention the missing curve, and the curve subplot
  must contain a placeholder text element.
- **(d) uppercase extension fallback** — regression for the MaIA dataset's
  lone uppercase case ``S_32.JPG``. A synthetic ``S_99.JPG`` is dropped
  into a temp Scoliosis/ directory and the CLI's case-loading helpers
  are exercised directly: ``_resolve_image_path`` must return the
  uppercase variant and ``_load_image_array`` must decode it into a 2-D
  ``uint8`` numpy array.

The tests use only the ``Agg`` backend (forced by importing the
visualization module), construct deterministic synthetic inputs entirely
in numpy, and never compare pixel-level golden images. They therefore
have no font, locale, or matplotlib-version dependency and are safe to
run on any developer machine or CI host.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.text import Text
from PIL import Image

from ai.visualization import (
    TARGET_VERTEBRA_IDS,
    CaseAudit,
    render_case_summary,
)
from scripts.build_case_summaries import (
    _load_image_array,
    _resolve_image_path,
)

# ──────────────────────────────────────────────────────────────────────────
# Synthetic input fixtures
# ──────────────────────────────────────────────────────────────────────────

#: Image dimensions used by every test in this module. Chosen to match the
#: cavekit R5 test (a) spec verbatim.
_IMG_H: int = 256
_IMG_W: int = 128


def _synthetic_image() -> np.ndarray:
    """Build a deterministic 256×128 grayscale gradient image."""

    return (
        np.linspace(0, 255, _IMG_H * _IMG_W, dtype=np.float32)
        .reshape(_IMG_H, _IMG_W)
        .astype(np.uint8)
    )


def _synthetic_full_mask() -> np.ndarray:
    """Multiclass mask with all 17 target vertebra IDs (6..22) present.

    Each vertebra is painted as a horizontal strip so the mask carries
    every target ID at least once and ``_present_target_ids`` returns the
    full list. Centroid columns vary linearly across the image so the
    derived Cobb angle does not raise a polynomial-fit singularity.
    """

    mask = np.zeros((_IMG_H, _IMG_W), dtype=np.uint8)
    n = len(TARGET_VERTEBRA_IDS)
    band = _IMG_H // (n + 2)
    for i, vid in enumerate(TARGET_VERTEBRA_IDS):
        y0 = (i + 1) * band
        y1 = y0 + band - 2
        # Slight horizontal drift so centroids form a curve, not a column.
        x_offset = int(8 * np.sin(i / 2.0))
        mask[y0:y1, 30 + x_offset : 90 + x_offset] = vid
    return mask


def _synthetic_curve() -> np.ndarray:
    """Two-column ``(x_px, y_px)`` polyline approximating the spine."""

    ys = np.linspace(10, _IMG_H - 10, 40, dtype=np.float32)
    xs = (_IMG_W / 2) + 6.0 * np.sin(np.linspace(0, 6.28, 40, dtype=np.float32))
    return np.column_stack([xs, ys])


def _has_text_element(ax) -> bool:
    """Return True iff the axis carries at least one ``matplotlib.text.Text``
    element with non-empty content. Tick labels are excluded so we only
    detect deliberate placeholder text drawn by the renderer.
    """

    deliberate_texts = [
        t for t in ax.texts
        if isinstance(t, Text) and t.get_text().strip()
    ]
    return len(deliberate_texts) > 0


# ──────────────────────────────────────────────────────────────────────────
# (a) Ok path
# ──────────────────────────────────────────────────────────────────────────


def test_render_ok_path_returns_ok_status_and_six_axes() -> None:
    """R5(a): full 17-vertebra mask + curve → ok, six axes, no warnings."""

    image = _synthetic_image()
    mask = _synthetic_full_mask()
    curve = _synthetic_curve()
    audit = CaseAudit(
        patient_id=1,
        category="Scoliosis",
        severity="warn",
        issue_codes=("low_target_class_count",),
        cobb_reported_deg=15.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, status = render_case_summary(image, mask, audit, curve=curve)

    assert status["render_status"] == "ok", status
    assert status["render_notes"] == "", status
    assert status["n_target_vertebrae"] == 17, status
    assert status["image_h"] == _IMG_H
    assert status["image_w"] == _IMG_W
    assert len(fig.axes) == 6, len(fig.axes)


# ──────────────────────────────────────────────────────────────────────────
# (b) Empty mask → partial
# ──────────────────────────────────────────────────────────────────────────


def test_render_empty_mask_downgrades_to_partial_with_placeholder() -> None:
    """R5(b): all-zero mask → partial, notes mention empty mask, placeholder."""

    image = _synthetic_image()
    mask = np.zeros_like(image, dtype=np.uint8)
    audit = CaseAudit(
        patient_id=2,
        category="Scoliosis",
        severity="fatal",
        issue_codes=("zero_target_classes",),
        cobb_reported_deg=30.0,
    )
    curve = _synthetic_curve()

    fig, status = render_case_summary(image, mask, audit, curve=curve)

    assert status["render_status"] == "partial", status
    assert status["n_target_vertebrae"] == 0
    assert len(status["missing_vertebrae"]) == 17
    assert "no target vertebrae" in status["render_notes"], status["render_notes"]
    # The mask subplot is index 1 in the 2x3 grid (row 0 / col 1).
    mask_ax = fig.axes[1]
    assert _has_text_element(mask_ax), (
        "mask subplot should carry a placeholder text element"
    )


# ──────────────────────────────────────────────────────────────────────────
# (c) Missing curve → partial
# ──────────────────────────────────────────────────────────────────────────


def test_render_missing_curve_downgrades_to_partial_with_placeholder() -> None:
    """R5(c): Normal case, non-empty mask, curve=None → partial + placeholder."""

    image = _synthetic_image()
    mask = _synthetic_full_mask()
    audit = CaseAudit(
        patient_id=3,
        category="Normal",
        severity="warn",
        issue_codes=(),
        cobb_reported_deg=None,
    )

    fig, status = render_case_summary(image, mask, audit, curve=None)

    assert status["render_status"] == "partial", status
    assert "no curve data" in status["render_notes"], status["render_notes"]
    assert status["cobb_reported_deg"] is None
    # The curve subplot is index 3 in the 2x3 grid (row 1 / col 0).
    curve_ax = fig.axes[3]
    assert _has_text_element(curve_ax), (
        "curve subplot should carry a placeholder text element"
    )


# ──────────────────────────────────────────────────────────────────────────
# (d) Extension fallback regression — uppercase ``S_99.JPG``
# ──────────────────────────────────────────────────────────────────────────


def test_cli_resolves_uppercase_extension_and_loads_image(
    tmp_path: Path,
) -> None:
    """R5(d): CLI loader resolves ``S_99.JPG`` via the uppercase fallback.

    The MaIA dataset ships exactly one case with an uppercase ``.JPG``
    extension (``Scoliosis/S_32.JPG``) while every other case is
    lowercase. A naive lowercase-only lookup would silently drop that
    case from the working set. This test plants a synthetic uppercase
    radiograph in a temp directory, calls ``_resolve_image_path`` (the
    CLI helper that R2 pins to the lowercase-then-uppercase fallback
    order) and confirms both that the uppercase variant is found and
    that ``_load_image_array`` decodes it into the expected shape.
    """

    # Arrange: temp raw dataset directory containing only ``S_99.JPG``.
    scoliosis_dir = tmp_path / "Scoliosis"
    scoliosis_dir.mkdir()
    uppercase_path = scoliosis_dir / "S_99.JPG"

    height, width = 64, 32
    # Deterministic gradient → decodes back cleanly even through JPEG
    # lossy compression because we only assert shape and dtype below,
    # not per-pixel equality.
    pixels = (
        np.linspace(0, 255, height * width, dtype=np.float32)
        .reshape(height, width)
        .astype(np.uint8)
    )
    Image.fromarray(pixels, mode="L").save(uppercase_path, format="JPEG")

    # Act: run the CLI's case-loading helpers exactly as
    # ``_render_one_case`` would for patient 99 / Scoliosis.
    resolved = _resolve_image_path(tmp_path, "Scoliosis", 99)

    # Assert: the fallback actually hit the uppercase variant, not a
    # phantom lowercase path.
    assert resolved is not None, (
        "_resolve_image_path should fall back to the uppercase .JPG"
    )
    assert resolved == uppercase_path
    assert resolved.suffix == ".JPG"

    # And the uppercase path decodes cleanly into a 2-D uint8 array.
    loaded = _load_image_array(resolved)
    assert loaded.shape == (height, width), loaded.shape
    assert loaded.dtype == np.uint8
