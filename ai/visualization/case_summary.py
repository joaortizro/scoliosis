"""Per-case visualization summary rendering.

Cavekit: cavekit-case-visualization.md R1.

This module provides ``render_case_summary``, a pure rendering function that
turns a single MaIA case (image, multiclass mask, audit metadata, optional
spinal curve, optional derived keypoints) into a six-subplot matplotlib
figure plus a structured status dictionary.

The function is intentionally I/O-free — the caller is responsible for
loading inputs from disk and persisting the returned figure. This keeps the
library importable from any context (notebook, CLI, future inference adapter)
without dragging in DVC, MLflow, FastAPI, or SQLAlchemy. The architectural
fitness check in ``tests/test_visualization_architecture.py`` enforces that
constraint.

Determinism contract
--------------------
``render_case_summary`` is deterministic: identical inputs always yield
identical figure pixels and identical status dictionaries. There are no
timestamps, no unseeded random calls, and no system-clock reads anywhere
in the module. This allows the calling DVC stage
(``scripts/build_case_summaries.py``) to produce a reproducible
``data/processed/case_summaries`` directory whose hash does not change
across reruns.

Never-raises contract
---------------------
The entire function body is wrapped in a top-level ``try/except`` so any
unexpected internal failure degrades gracefully to
``render_status = "failed"`` with a populated ``render_notes`` field, rather
than propagating the exception. Per-subplot defensive wrapping (so a single
broken panel does not poison the rest of the figure) is layered on top in
T-102.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib

# Force a non-interactive backend so the module is safe to import in headless
# CLI / DVC stage environments. Callers that want a different backend should
# set it before importing this module.
matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ai.evaluation.cobb import (
    cobb_from_raw_multiclass_mask,
    cobb_from_raw_multiclass_mask_endplates,
)

# ──────────────────────────────────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────────────────────────────────

#: Raw vertebra IDs in the multiclass mask that count as "target" (T1..L5).
TARGET_VERTEBRA_IDS: tuple[int, ...] = tuple(range(6, 23))  # 6..22 inclusive

#: Anatomical names for the 17 target vertebrae, in head-to-tail order. The
#: index of a name in this tuple matches the index of its raw ID in
#: ``TARGET_VERTEBRA_IDS``.
TARGET_VERTEBRA_NAMES: tuple[str, ...] = (
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
)

#: Closed set of keys the status dict returned by ``render_case_summary``
#: contains. ``tests/test_case_summary.py`` enforces this exactly.
_STATUS_KEYS: tuple[str, ...] = (
    "render_status",
    "render_notes",
    "n_target_vertebrae",
    "missing_vertebrae",
    "image_h",
    "image_w",
    "aspect_ratio",
    "cobb_reported_deg",
    "cobb_derived_deg",
    "cobb_delta_deg",
    "cobb_endplates_deg",
    "cobb_endplates_delta_deg",
)


# ──────────────────────────────────────────────────────────────────────────
# Audit metadata record
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseAudit:
    """Per-case audit metadata consumed by ``render_case_summary``.

    Constructed by the CLI from ``known_issues.csv`` and ``clean_index.csv``
    rows for a single case. Frozen so it is hashable and cannot drift after
    construction.

    Attributes:
        patient_id: Integer patient ID. Used for the panel title and the
            output PNG filename (zero-padded to 4 digits).
        category: ``"Scoliosis"`` or ``"Normal"``.
        severity: Highest severity bucket among the case's flagged issues —
            ``"warn"`` or ``"fatal"``. ``"info"`` is excluded from the
            working set, so it never appears here.
        issue_codes: Issue codes for the case, in arbitrary order. The CLI
            sorts and filters these (excluding ``id_out_of_range``) when
            writing the index CSV; the renderer just displays them in the
            text block.
        cobb_reported_deg: Cobb angle from the MaIA metrics JSON for
            Scoliosis cases, ``None`` for Normal cases or when the metrics
            JSON is missing.
    """

    patient_id: int
    category: str
    severity: str
    issue_codes: tuple[str, ...] = field(default_factory=tuple)
    cobb_reported_deg: float | None = None


# ──────────────────────────────────────────────────────────────────────────
# Default status dict factory
# ──────────────────────────────────────────────────────────────────────────


def _default_status() -> dict[str, Any]:
    """Return a fresh status dict with the closed key set and safe defaults.

    A failed render returns this with ``render_status="failed"`` and a
    populated ``render_notes`` — so the caller can always trust the dict
    has the same shape regardless of whether rendering succeeded.
    """

    return {
        "render_status": "ok",
        "render_notes": "",
        "n_target_vertebrae": 0,
        "missing_vertebrae": list(TARGET_VERTEBRA_NAMES),
        "image_h": 0,
        "image_w": 0,
        "aspect_ratio": 0.0,
        "cobb_reported_deg": None,
        "cobb_derived_deg": None,
        "cobb_delta_deg": None,
        "cobb_endplates_deg": None,
        "cobb_endplates_delta_deg": None,
    }


# ──────────────────────────────────────────────────────────────────────────
# Status field computation (no figure side-effects)
# ──────────────────────────────────────────────────────────────────────────


def _present_target_ids(mask: np.ndarray) -> tuple[int, ...]:
    """Return the subset of ``TARGET_VERTEBRA_IDS`` actually present in mask.

    Order matches anatomical order (head → tail). A vertebra "is present"
    if at least one pixel of the mask carries its ID.
    """

    present: list[int] = []
    for vid in TARGET_VERTEBRA_IDS:
        if np.any(mask == vid):
            present.append(vid)
    return tuple(present)


def _missing_vertebra_names(present_ids: Sequence[int]) -> list[str]:
    """Return anatomical names of target vertebrae absent from the mask."""

    present_set = set(present_ids)
    return [
        name
        for vid, name in zip(TARGET_VERTEBRA_IDS, TARGET_VERTEBRA_NAMES, strict=True)
        if vid not in present_set
    ]


#: Minimum number of distinct target vertebrae required to fit a curve and
#: extract a meaningful Cobb angle. Mirrors the polynomial-degree guard
#: inside ``ai.evaluation.cobb._cobb_from_centroids``.
_MIN_VERTEBRAE_FOR_COBB: int = 5


def _safe_cobb_from_mask(
    mask: np.ndarray,
    n_present_targets: int,
) -> float | None:
    """Try to derive a Cobb angle from the raw multiclass mask.

    Returns ``None`` (rather than raising) if derivation cannot produce a
    meaningful angle: empty/sparse mask (fewer than 5 target vertebrae),
    polynomial fit singularity, or unexpected exception. A successful
    derivation always returns a finite ``float``.
    """

    if n_present_targets < _MIN_VERTEBRAE_FOR_COBB:
        return None
    try:
        value = cobb_from_raw_multiclass_mask(mask)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _safe_cobb_endplates_from_mask(
    mask: np.ndarray,
    n_present_targets: int,
) -> float | None:
    """Try to derive a Cobb angle via the per-vertebra endplate-tilt method.

    Companion to ``_safe_cobb_from_mask`` that wraps
    ``cobb_from_raw_multiclass_mask_endplates`` (AASCE/BoostNet-style
    per-vertebra PCA). Returns ``None`` under the same degeneracy
    conditions as the centroid method so the two columns line up in
    ``index.csv``.
    """

    if n_present_targets < _MIN_VERTEBRAE_FOR_COBB:
        return None
    try:
        value = cobb_from_raw_multiclass_mask_endplates(mask)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _populate_data_fields(
    status: dict[str, Any],
    image: np.ndarray,
    mask: np.ndarray,
    audit: CaseAudit,
) -> None:
    """Fill the non-render fields of the status dict from the inputs.

    Mutates ``status`` in place. The render-time fields (``render_status``,
    ``render_notes``) are managed by the subplot pipeline (T-102) and the
    top-level wrapper.
    """

    image_h = int(image.shape[0]) if image.ndim >= 2 else 0
    image_w = int(image.shape[1]) if image.ndim >= 2 else 0
    status["image_h"] = image_h
    status["image_w"] = image_w
    status["aspect_ratio"] = round(image_h / image_w, 4) if image_w > 0 else 0.0

    present_ids = _present_target_ids(mask)
    status["n_target_vertebrae"] = len(present_ids)
    status["missing_vertebrae"] = _missing_vertebra_names(present_ids)

    reported = audit.cobb_reported_deg
    derived = _safe_cobb_from_mask(mask, len(present_ids))
    endplates = _safe_cobb_endplates_from_mask(mask, len(present_ids))
    status["cobb_reported_deg"] = float(reported) if reported is not None else None
    status["cobb_derived_deg"] = derived
    status["cobb_endplates_deg"] = endplates
    if reported is not None and derived is not None:
        status["cobb_delta_deg"] = round(derived - float(reported), 4)
    else:
        status["cobb_delta_deg"] = None
    if reported is not None and endplates is not None:
        status["cobb_endplates_delta_deg"] = round(
            endplates - float(reported), 4,
        )
    else:
        status["cobb_endplates_delta_deg"] = None


# ──────────────────────────────────────────────────────────────────────────
# Figure scaffold
# ──────────────────────────────────────────────────────────────────────────

#: Fixed figure dimensions in inches. Hard-coded so PNG output is byte-stable
#: across runs (R2 acceptance criterion: deterministic dpi/figure dimensions).
_FIG_WIDTH_IN: float = 14.0
_FIG_HEIGHT_IN: float = 9.0
_FIG_DPI: int = 130


def _build_empty_figure() -> tuple[Figure, list[plt.Axes]]:
    """Create the six-axes scaffold figure used by the renderer.

    The layout is a 2x3 grid (top row: raw / mask / keypoints; bottom row:
    curve / text / coverage bar). T-101 returns this scaffold with empty
    axes; T-102 populates each panel with real content.
    """

    fig = plt.figure(figsize=(_FIG_WIDTH_IN, _FIG_HEIGHT_IN), dpi=_FIG_DPI)
    gs = fig.add_gridspec(
        2, 3,
        hspace=0.25,
        wspace=0.15,
        left=0.04,
        right=0.97,
        top=0.93,
        bottom=0.05,
    )
    axes: list[plt.Axes] = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────
# Per-subplot drawers
# ──────────────────────────────────────────────────────────────────────────


def _to_2d_gray(image: np.ndarray) -> np.ndarray:
    """Coerce a 2-D or 3-D image to a 2-D grayscale view (no copy when 2-D).

    Multi-channel inputs are reduced to the first channel — the renderer
    only displays grayscale, so we never need to honour colour data.
    """

    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[..., 0]
    raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}")


def _strip_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_placeholder(ax: plt.Axes, message: str) -> None:
    """Render a centred grey notice on an otherwise empty axis."""

    _strip_ticks(ax)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=10, color="#777777",
    )


def _draw_raw_image(
    ax: plt.Axes, image: np.ndarray, audit: CaseAudit,
) -> None:
    """Subplot 0 — raw radiograph in grayscale."""

    img2d = _to_2d_gray(image)
    ax.imshow(img2d, cmap="gray")
    ax.set_title(
        f"Raw — {audit.category[0]}_{audit.patient_id:04d}", fontsize=10,
    )
    _strip_ticks(ax)


def _draw_mask_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    has_targets: bool,
) -> None:
    """Subplot 1 — faint image + tab20 multiclass overlay restricted to T1–L5."""

    img2d = _to_2d_gray(image)
    ax.imshow(img2d, cmap="gray", alpha=0.35)
    ax.set_title("Mask overlay (T1–L5)", fontsize=10)
    if not has_targets:
        _draw_placeholder(ax, "no target vertebrae in mask")
        return
    lo, hi = TARGET_VERTEBRA_IDS[0], TARGET_VERTEBRA_IDS[-1]
    masked = np.ma.masked_where((mask < lo) | (mask > hi), mask)
    ax.imshow(masked, cmap="tab20", vmin=0, vmax=35, alpha=0.65)
    _strip_ticks(ax)


def _draw_keypoints(
    ax: plt.Axes,
    image: np.ndarray,
    keypoints: np.ndarray | None,
    has_targets: bool,
) -> None:
    """Subplot 2 — faint image + scatter of derived endplate keypoints."""

    img2d = _to_2d_gray(image)
    ax.imshow(img2d, cmap="gray", alpha=0.35)
    ax.set_title("Keypoints", fontsize=10)
    if not has_targets:
        _draw_placeholder(ax, "no target vertebrae")
        return
    if keypoints is None:
        _draw_placeholder(ax, "no keypoints provided")
        return
    pts = np.asarray(keypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        _draw_placeholder(ax, "keypoints malformed")
        return
    valid = (
        np.isfinite(pts).all(axis=-1)
        & (pts[:, 0] >= 0.0)
        & (pts[:, 1] >= 0.0)
    )
    if not bool(valid.any()):
        _draw_placeholder(ax, "no valid keypoints")
        return
    ax.scatter(
        pts[valid, 0], pts[valid, 1],
        s=6, c="#33ff66", marker=".",
        edgecolors="black", linewidths=0.2,
    )
    ax.set_title(f"Keypoints (n={int(valid.sum())})", fontsize=10)
    _strip_ticks(ax)


def _draw_curve(
    ax: plt.Axes, image: np.ndarray, curve: np.ndarray | None,
) -> None:
    """Subplot 3 — faint image + MaIA spinal-curve polyline."""

    img2d = _to_2d_gray(image)
    ax.imshow(img2d, cmap="gray", alpha=0.35)
    ax.set_title("Spinal curve", fontsize=10)
    if curve is None:
        _draw_placeholder(ax, "no curve data")
        return
    pts = np.asarray(curve, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        _draw_placeholder(ax, "curve malformed")
        return
    ax.plot(pts[:, 0], pts[:, 1], color="#ff9933", linewidth=1.6)
    ax.set_title(f"Spinal curve (n={pts.shape[0]})", fontsize=10)
    _strip_ticks(ax)


def _format_cobb(value: float | None) -> str:
    return f"{value:6.1f}°" if value is not None else "    —  "


def _format_cobb_delta(value: float | None) -> str:
    return f"{value:+6.1f}°" if value is not None else "    —  "


def _draw_text_block(
    ax: plt.Axes, audit: CaseAudit, status: dict[str, Any],
) -> None:
    """Subplot 4 — text-only audit summary panel."""

    _strip_ticks(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)

    lines: list[str] = [
        f"Patient   : {audit.patient_id:04d}",
        f"Category  : {audit.category}",
        f"Severity  : {audit.severity}",
        "",
        "Issue codes:",
    ]
    if audit.issue_codes:
        for code in audit.issue_codes:
            lines.append(f"  • {code}")
    else:
        lines.append("  (none)")
    lines += [
        "",
        f"Cobb reported : {_format_cobb(status['cobb_reported_deg'])}",
        f"Cobb derived  : {_format_cobb(status['cobb_derived_deg'])}",
        f"Cobb delta    : {_format_cobb_delta(status['cobb_delta_deg'])}",
        f"Cobb endplates: {_format_cobb(status['cobb_endplates_deg'])}",
        f"Cobb ep delta : {_format_cobb_delta(status['cobb_endplates_delta_deg'])}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines),
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize=9, family="monospace",
    )


def _draw_coverage_bar(ax: plt.Axes, status: dict[str, Any]) -> None:
    """Subplot 5 — T1–L5 coverage bar (green=present, red=missing)."""

    n = len(TARGET_VERTEBRA_NAMES)
    missing_set = set(status["missing_vertebrae"])
    colors = [
        "#cc3333" if name in missing_set else "#33aa55"
        for name in TARGET_VERTEBRA_NAMES
    ]
    xs = np.arange(n)
    ax.bar(
        xs, np.ones(n),
        color=colors, edgecolor="black", linewidth=0.4, width=0.85,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(TARGET_VERTEBRA_NAMES, fontsize=7, rotation=60)
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.1)
    ax.set_title(
        f"Coverage: {status['n_target_vertebrae']}/{n}", fontsize=10,
    )
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────
# Subplot orchestrator (per-panel defensive wrapping)
# ──────────────────────────────────────────────────────────────────────────


def _render_panels(
    fig: Figure,
    axes: Sequence[plt.Axes],
    image: np.ndarray,
    mask: np.ndarray,
    audit: CaseAudit,
    curve: np.ndarray | None,
    keypoints: np.ndarray | None,
    status: dict[str, Any],
) -> list[str]:
    """Draw all six subplots defensively. Returns degradation notes.

    Each panel is wrapped in its own try/except so a single broken panel
    cannot poison the rest of the figure. Soft-degradation cases (no
    target vertebrae, missing curve for a Scoliosis case) also append
    notes so the caller downgrades ``render_status`` to ``"partial"``.
    """

    notes: list[str] = []
    has_targets = status["n_target_vertebrae"] > 0

    panels = (
        ("raw",       lambda ax: _draw_raw_image(ax, image, audit)),
        ("mask",      lambda ax: _draw_mask_overlay(ax, image, mask, has_targets)),
        ("keypoints", lambda ax: _draw_keypoints(ax, image, keypoints, has_targets)),
        ("curve",     lambda ax: _draw_curve(ax, image, curve)),
        ("text",      lambda ax: _draw_text_block(ax, audit, status)),
        ("coverage",  lambda ax: _draw_coverage_bar(ax, status)),
    )

    for ax, (name, drawer) in zip(axes, panels, strict=True):
        try:
            drawer(ax)
        except Exception as exc:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")
            try:
                ax.clear()
                _draw_placeholder(ax, f"{name} failed")
            except Exception:
                pass

    # Soft-degradation: panels that fell back to placeholders for missing
    # input rather than crashing. Per cavekit-case-visualization R1, both
    # an empty target mask and a missing MaIA curve downgrade the render
    # to "partial" so the index.csv flags every case where one or more
    # panels are showing a placeholder rather than real data.
    if not has_targets:
        notes.append("no target vertebrae present")
    if curve is None:
        notes.append("no curve data")

    fig.suptitle(
        f"Case audit — {audit.category[0]}_{audit.patient_id:04d}"
        f" ({audit.severity})",
        fontsize=12,
    )

    return notes


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def render_case_summary(
    image: np.ndarray,
    mask: np.ndarray,
    audit: CaseAudit,
    curve: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Render a per-case summary panel and return ``(figure, status)``.

    Pure rendering — no file I/O, deterministic, never raises. See module
    docstring for the determinism and never-raises contracts.

    Args:
        image: 2-D ``(H, W)`` ndarray of the raw radiograph (any numeric
            dtype). Multi-channel images are accepted but only the first
            channel is rendered.
        mask: 2-D ``(H, W)`` ndarray of raw multiclass IDs (vertebra IDs in
            the 1..35 source space). Values outside the target range
            ``6..22`` are ignored when computing target coverage and the
            derived Cobb angle.
        audit: ``CaseAudit`` metadata record for this case.
        curve: Optional ``(N, 2)`` ndarray of MaIA spinal-curve points in
            ``(x_px, y_px)`` order. ``None`` for Normal cases or cases
            whose ``curve_csv`` file is missing.
        keypoints: Optional ``(68, 2)`` ndarray of derived endplate corner
            keypoints in canonical TL/TR/BL/BR order (see
            ``ai.preprocessing.keypoints``). ``None`` if the caller did not
            derive keypoints for this case.

    Returns:
        Tuple ``(figure, status)`` where ``figure`` is a matplotlib
        ``Figure`` containing six subplots and ``status`` is a dict with
        exactly the keys: ``render_status``, ``render_notes``,
        ``n_target_vertebrae``, ``missing_vertebrae``, ``image_h``,
        ``image_w``, ``aspect_ratio``, ``cobb_reported_deg``,
        ``cobb_derived_deg``, ``cobb_delta_deg``,
        ``cobb_endplates_deg``, ``cobb_endplates_delta_deg``.

        ``render_status`` is one of ``"ok"``, ``"partial"``, ``"failed"``.
        On ``"failed"``, the figure is still a valid (possibly empty)
        ``Figure`` and ``render_notes`` describes the failure.
    """

    # Build the scaffold figure FIRST so even a catastrophic failure during
    # data extraction still returns a valid Figure object.
    try:
        fig, axes = _build_empty_figure()
    except Exception as exc:  # pragma: no cover — matplotlib init should not fail
        fig = plt.figure(figsize=(_FIG_WIDTH_IN, _FIG_HEIGHT_IN), dpi=_FIG_DPI)
        status = _default_status()
        status["render_status"] = "failed"
        status["render_notes"] = f"figure init error: {type(exc).__name__}: {exc}"
        return fig, status

    status = _default_status()

    try:
        _populate_data_fields(status, image, mask, audit)
        notes = _render_panels(
            fig, axes, image, mask, audit, curve, keypoints, status,
        )
        if notes:
            status["render_status"] = "partial"
            status["render_notes"] = "; ".join(notes)
    except Exception as exc:
        status["render_status"] = "failed"
        status["render_notes"] = f"render error: {type(exc).__name__}: {exc}"

    # Sanity check: the status dict must have exactly the closed key set.
    # Done inside the function so a future edit that drifts the schema is
    # caught immediately by the unit tests.
    assert set(status.keys()) == set(_STATUS_KEYS), (
        f"status dict drift: {sorted(status.keys())} != {sorted(_STATUS_KEYS)}"
    )

    return fig, status
