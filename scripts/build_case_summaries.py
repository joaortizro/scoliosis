"""Build per-case visualization summaries for the MaIA audit working set.

Cavekit: cavekit-case-visualization.md R2. Build-site tasks: T-103 (this
CLI shell), T-104 (per-case rendering loop with extension fallback),
T-105 (frozen 15-column index.csv writer), T-106 (failures.txt sidecar
and end-of-run summary). Wired into the DVC pipeline as the
``case_summaries`` stage by T-107.

The script reads ``data/processed/audit/known_issues.csv`` plus
``data/processed/audit/clean_index.csv``, builds a deterministic working
set of every case at ``warn`` or ``fatal`` severity, calls
``ai.visualization.render_case_summary`` per case (delegated to T-104),
and writes one PNG plus one row of the frozen 15-column ``index.csv``
schema per case to ``data/processed/case_summaries/`` (delegated to
T-105). When at least one case fails, a sidecar ``failures.txt`` is
written next to the index (delegated to T-106).

Skip-and-log contract
---------------------
The script always exits 0 under R2: per-case crashes are captured in
``render_status="failed"`` rows of the index and never propagate. The
only non-zero exits are reserved for missing inputs (the audit CSVs or
the raw dataset directory) and uncaught programming errors that escape
the per-case loop.

Path resolution
---------------
Every path is resolved relative to the repository root (derived from
``__file__``), not the current working directory, so the script behaves
identically when invoked from ``scripts/``, the repo root, or anywhere
else (e.g. inside a DVC stage subprocess).
"""

import csv
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Path bootstrap
# ──────────────────────────────────────────────────────────────────────────
#
# Make ``ai`` importable when this file is invoked directly via
# ``python scripts/build_case_summaries.py``. Without this bootstrap
# Python only puts the script's own directory (``scripts/``) on
# ``sys.path``, so the top-level ``ai`` package is invisible. Inserting
# the repo root unconditionally is safe and idempotent — it is a no-op
# when the package has already been installed via ``pip install -e .``.

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402  (must follow path bootstrap)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ai.preprocessing.keypoints import multiclass_mask_to_keypoints  # noqa: E402
from ai.visualization import CaseAudit, render_case_summary  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Repo-root-anchored paths
# ──────────────────────────────────────────────────────────────────────────
# ``REPO_ROOT`` is defined above in the path bootstrap so the ``ai``
# imports below it can succeed. The remaining input/output paths are
# derived from it here.

KNOWN_ISSUES_CSV: Path = (
    REPO_ROOT / "data" / "processed" / "audit" / "known_issues.csv"
)
CLEAN_INDEX_CSV: Path = (
    REPO_ROOT / "data" / "processed" / "audit" / "clean_index.csv"
)
RAW_DATASET_DIR: Path = REPO_ROOT / "data" / "raw" / "MaIA_Scoliosis_Dataset"
OUTPUT_DIR: Path = REPO_ROOT / "data" / "processed" / "case_summaries"
INDEX_CSV: Path = OUTPUT_DIR / "index.csv"
FAILURES_TXT: Path = OUTPUT_DIR / "failures.txt"

#: Frozen column order for ``index.csv`` per cavekit R3. The script
#: never recomputes these values — every column is read directly from
#: the status dict returned by ``render_case_summary`` (R2 acceptance
#: criterion "does not recompute missing_vertebrae, image_h, ..." ).
_INDEX_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "category",
    "severity",
    "flags",
    "render_status",
    "render_notes",
    "n_target_vertebrae",
    "missing_vertebrae",
    "cobb_reported_deg",
    "cobb_derived_deg",
    "cobb_delta_deg",
    "cobb_endplates_deg",
    "cobb_endplates_delta_deg",
    "image_h",
    "image_w",
    "aspect_ratio",
    "png_path",
)

#: Severities promoted into the working set. ``info``-level issues are
#: dataset-wide annotations (such as ``id_out_of_range``) that do not
#: warrant per-case visual review and are filtered out here.
WORKING_SEVERITIES: frozenset[str] = frozenset({"warn", "fatal"})

#: Ordered severity rank used to pick the worst severity across a case's
#: per-row issues. Mirrors the canonical audit severity ordering.
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warn": 1, "fatal": 2}

#: Filename extensions to try when resolving a raw radiograph path.
#: Lowercase variants are tried first because they are the dataset norm;
#: uppercase variants exist as a regression for ``S_32.JPG`` (the only
#: case in the MaIA dataset that ships with an uppercase extension).
_FALLBACK_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG",
)


# ──────────────────────────────────────────────────────────────────────────
# Working-set construction
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseKey:
    """Identity of a single MaIA case — patient ID + category."""

    patient_id: int
    category: str


@dataclass(frozen=True)
class WorkingCase:
    """One entry in the deterministic warn+fatal working set."""

    key: CaseKey
    severity: str
    issue_codes: tuple[str, ...]


def _load_known_issues(
    path: Path,
) -> dict[CaseKey, tuple[str, tuple[str, ...]]]:
    """Aggregate ``known_issues.csv`` rows by (patient_id, category).

    Returns a mapping of ``CaseKey`` → ``(worst_severity, issue_codes)``.
    Severity is the maximum of the per-row severities for that case.
    Issue codes are emitted in CSV row order with duplicates preserved —
    T-105 is responsible for de-duplicating, sorting, and filtering
    ``id_out_of_range`` when writing the frozen index column.
    """

    codes_by_case: dict[CaseKey, list[str]] = defaultdict(list)
    severities: dict[CaseKey, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_pid = row.get("patient_id", "")
            if not raw_pid or not raw_pid.strip():
                # Issues not tied to a specific case (e.g. dataset-wide
                # structural warnings) are skipped from the working set.
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            key = CaseKey(patient_id=pid, category=row["category"])
            sev = row["severity"]
            codes_by_case[key].append(row["issue_code"])
            prev = severities.get(key)
            if prev is None or _SEVERITY_RANK[sev] > _SEVERITY_RANK[prev]:
                severities[key] = sev
    return {
        key: (severities[key], tuple(codes_by_case[key]))
        for key in codes_by_case
    }


def _load_clean_index(path: Path) -> dict[CaseKey, dict[str, str]]:
    """Read ``clean_index.csv`` keyed by (patient_id, category).

    Returns a mapping of ``CaseKey`` → original CSV row dict so the
    per-case loop (T-104) can pull image/mask/metrics paths without
    re-parsing the CSV.
    """

    out: dict[CaseKey, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_pid = row.get("patient_id", "")
            if not raw_pid or not raw_pid.strip():
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            key = CaseKey(patient_id=pid, category=row["category"])
            out[key] = dict(row)
    return out


def _build_working_set(
    known_issues: dict[CaseKey, tuple[str, tuple[str, ...]]],
) -> list[WorkingCase]:
    """Build the warn+fatal working set in deterministic sort order.

    Cases are sorted by ``(category, patient_id)`` so iteration order is
    stable across runs and platforms — the produced ``index.csv`` row
    order matches by construction (R3 acceptance criterion).
    """

    cases: list[WorkingCase] = []
    for key, (severity, codes) in known_issues.items():
        if severity not in WORKING_SEVERITIES:
            continue
        cases.append(
            WorkingCase(key=key, severity=severity, issue_codes=codes),
        )
    cases.sort(key=lambda wc: (wc.key.category, wc.key.patient_id))
    return cases


# ──────────────────────────────────────────────────────────────────────────
# Per-case input loaders
# ──────────────────────────────────────────────────────────────────────────


def _category_prefix(category: str) -> str:
    """Single-letter prefix used in dataset filenames (``S`` or ``N``)."""

    return category[0] if category else "X"


def _resolve_image_path(
    raw_dir: Path, category: str, patient_id: int,
) -> Path | None:
    """Resolve a raw radiograph path with extension fallback.

    Tries lowercase ``.jpg``/``.jpeg``/``.png`` first, then their
    uppercase variants in the same order. Returns ``None`` when none of
    the candidates exist on disk. The fallback is the regression covered
    by build-site task T-109 — the MaIA dataset ships ``S_32.JPG`` with
    an uppercase extension while every other case is lowercase.
    """

    prefix = _category_prefix(category)
    stem = f"{prefix}_{patient_id}"
    case_dir = raw_dir / category
    for ext in _FALLBACK_EXTENSIONS:
        candidate = case_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_image_array(path: Path) -> np.ndarray:
    """Load a radiograph as a 2-D ``uint8`` array (first channel only)."""

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _load_mask_array(path: Path) -> np.ndarray:
    """Load a multiclass mask PNG as a 2-D integer array.

    The MaIA ``LabelMultiClass_ID_PNG`` masks store vertebra IDs in the
    1..35 range, so ``uint8`` is sufficient. Mode-``P`` palette PNGs are
    converted to ``L`` to expose the underlying index values.
    """

    with Image.open(path) as img:
        if img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img)
    return arr.astype(np.uint8, copy=False)


def _load_curve(path: Path | None) -> np.ndarray | None:
    """Load a MaIA spinal-curve CSV as an ``(N, 2)`` ``(x_px, y_px)`` array.

    Returns ``None`` when ``path`` is ``None`` (Normal case), the path is
    an empty string, the file does not exist, or the parsed array does
    not have the expected shape. Errors during parsing are swallowed and
    surface as a missing curve so the renderer can downgrade to
    ``partial`` rather than crashing.
    """

    if path is None:
        return None
    if not path.exists():
        return None
    try:
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = [(float(r["x_px"]), float(r["y_px"])) for r in reader]
    except (OSError, ValueError, KeyError):
        return None
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _load_reported_cobb_deg(path: Path | None) -> float | None:
    """Read ``cobb_angle_deg`` from a MaIA per-case metrics JSON.

    Returns ``None`` when ``path`` is ``None``, the file does not exist,
    the JSON cannot be parsed, or the field is absent. Normal cases have
    no metrics JSON and therefore always return ``None``.
    """

    if path is None or not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("cobb_angle_deg")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# Per-case render
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RenderResult:
    """One row of accumulated per-case render output.

    The CLI loop appends one of these per working-set case. T-105 reads
    ``status`` and ``png_path`` to populate the frozen 15-column index
    CSV; T-106 reads ``status['render_status']`` for the failures
    sidecar and the end-of-run summary tally.
    """

    case: WorkingCase
    status: dict[str, Any]
    png_path: str | None  # repo-root-relative POSIX path or None on failure


def _failed_status(message: str) -> dict[str, Any]:
    """Return a closed-schema status dict marked as ``failed``.

    Used when the per-case loader cannot even reach
    ``render_case_summary`` — for example because the case is missing
    from clean_index.csv or the image file cannot be resolved.
    """

    return {
        "render_status": "failed",
        "render_notes": message,
        "n_target_vertebrae": 0,
        "missing_vertebrae": [],
        "image_h": 0,
        "image_w": 0,
        "aspect_ratio": 0.0,
        "cobb_reported_deg": None,
        "cobb_derived_deg": None,
        "cobb_delta_deg": None,
        "cobb_endplates_deg": None,
        "cobb_endplates_delta_deg": None,
    }


def _filtered_issue_codes(codes: tuple[str, ...]) -> tuple[str, ...]:
    """Drop ``id_out_of_range`` (info-level) and de-dup, preserving order.

    The dataset-wide ``id_out_of_range`` annotation is excluded from the
    case-level audit text per cavekit R3. T-105 applies the same filter
    to the index ``flags`` column; doing it here too keeps the on-figure
    text block consistent with the index row.
    """

    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        if code == "id_out_of_range" or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return tuple(out)


def _render_one_case(
    case: WorkingCase,
    clean_row: dict[str, str] | None,
    raw_dir: Path,
    output_dir: Path,
) -> RenderResult:
    """Render a single working-set case end-to-end.

    Steps:
        1. Resolve and load the raw radiograph (with extension fallback).
        2. Load the multiclass mask from clean_index's recorded path.
        3. Load the optional MaIA curve and reported Cobb angle.
        4. Derive 68 endplate keypoints from the mask.
        5. Build the audit metadata record.
        6. Call ``render_case_summary``.
        7. Save the figure to ``{output_dir}/{prefix}_{patient_id:04d}.png``.
        8. Always close the figure to avoid leaking matplotlib state.

    Returns a ``RenderResult``. Per-case failures populate the
    ``status`` field with a ``failed`` row but never raise — the caller
    relies on this contract for the R2 skip-and-log invariant.
    """

    prefix = _category_prefix(case.key.category)
    png_name = f"{prefix}_{case.key.patient_id:04d}.png"
    png_abs = output_dir / png_name
    png_rel = png_abs.relative_to(REPO_ROOT).as_posix()

    if clean_row is None:
        return RenderResult(
            case=case,
            status=_failed_status("missing from clean_index.csv"),
            png_path=None,
        )

    image_path = _resolve_image_path(
        raw_dir, case.key.category, case.key.patient_id,
    )
    if image_path is None:
        return RenderResult(
            case=case,
            status=_failed_status(
                f"image not found for {prefix}_{case.key.patient_id:04d}"
            ),
            png_path=None,
        )

    mask_path_str = clean_row.get("multiclass_mask_path", "")
    mask_path = Path(mask_path_str) if mask_path_str else None
    if mask_path is None or not mask_path.exists():
        return RenderResult(
            case=case,
            status=_failed_status("multiclass mask not found"),
            png_path=None,
        )

    try:
        image = _load_image_array(image_path)
        mask = _load_mask_array(mask_path)
    except (OSError, ValueError) as exc:
        return RenderResult(
            case=case,
            status=_failed_status(
                f"input load error: {type(exc).__name__}: {exc}"
            ),
            png_path=None,
        )

    curve_path_str = clean_row.get("curve_csv_path", "")
    curve_path = Path(curve_path_str) if curve_path_str else None
    curve = _load_curve(curve_path)

    metrics_path_str = clean_row.get("metrics_json_path", "")
    metrics_path = Path(metrics_path_str) if metrics_path_str else None
    cobb_reported = _load_reported_cobb_deg(metrics_path)

    try:
        keypoints = multiclass_mask_to_keypoints(mask)
    except Exception:
        # Keypoint derivation is best-effort. Failures fall back to None
        # so the keypoints subplot draws a placeholder rather than the
        # whole case being marked failed.
        keypoints = None

    audit = CaseAudit(
        patient_id=case.key.patient_id,
        category=case.key.category,
        severity=case.severity,
        issue_codes=_filtered_issue_codes(case.issue_codes),
        cobb_reported_deg=cobb_reported,
    )

    fig, status = render_case_summary(
        image, mask, audit, curve=curve, keypoints=keypoints,
    )
    try:
        fig.savefig(png_abs, format="png")
    except Exception as exc:
        plt.close(fig)
        return RenderResult(
            case=case,
            status=_failed_status(
                f"savefig error: {type(exc).__name__}: {exc}"
            ),
            png_path=None,
        )
    plt.close(fig)

    return RenderResult(
        case=case,
        status=status,
        png_path=png_rel,
    )


# ──────────────────────────────────────────────────────────────────────────
# Frozen index CSV writer (R3)
# ──────────────────────────────────────────────────────────────────────────


def _format_flags_for_index(codes: tuple[str, ...]) -> str:
    """Format issue codes for the index ``flags`` column.

    Per R3: semicolon-separated, sorted alphabetically, ``id_out_of_range``
    excluded. Duplicates are removed. The empty-string case is used when
    a case has no codes left after filtering (should not happen in
    practice because the working set is warn+fatal only, but the
    failed-case path can produce an empty row).
    """

    kept = {c for c in codes if c and c != "id_out_of_range"}
    return ";".join(sorted(kept))


def _format_missing_vertebrae(names: list[str]) -> str:
    """Format the ``missing_vertebrae`` status field for the index.

    The status dict already carries the names in anatomical order
    (T1..T12, L1..L5). R3 preserves that order verbatim; the empty list
    becomes an empty string (i.e. "all 17 present").
    """

    if not names:
        return ""
    return ";".join(names)


def _format_optional_float(value: float | None) -> str:
    """Format a nullable Cobb field for the index.

    ``None`` renders as an empty string (R3: "empty when derivation
    fails" and "empty for Normal cases"). Concrete values are rendered
    via ``str`` so the on-disk representation preserves whatever
    precision the status dict carries — the script never re-rounds.
    """

    if value is None:
        return ""
    return str(value)


def _build_index_row(result: RenderResult) -> dict[str, str]:
    """Project a single ``RenderResult`` onto the frozen R3 schema."""

    case = result.case
    status = result.status
    return {
        "patient_id": str(case.key.patient_id),
        "category": case.key.category,
        "severity": case.severity,
        "flags": _format_flags_for_index(case.issue_codes),
        "render_status": str(status["render_status"]),
        "render_notes": str(status["render_notes"]),
        "n_target_vertebrae": str(status["n_target_vertebrae"]),
        "missing_vertebrae": _format_missing_vertebrae(
            list(status["missing_vertebrae"])
        ),
        "cobb_reported_deg": _format_optional_float(status["cobb_reported_deg"]),
        "cobb_derived_deg": _format_optional_float(status["cobb_derived_deg"]),
        "cobb_delta_deg": _format_optional_float(status["cobb_delta_deg"]),
        "cobb_endplates_deg": _format_optional_float(status["cobb_endplates_deg"]),
        "cobb_endplates_delta_deg": _format_optional_float(
            status["cobb_endplates_delta_deg"]
        ),
        "image_h": str(status["image_h"]),
        "image_w": str(status["image_w"]),
        "aspect_ratio": str(status["aspect_ratio"]),
        "png_path": result.png_path or "",
    }


def _write_index_csv(results: list[RenderResult], path: Path) -> None:
    """Write the frozen 15-column ``index.csv`` to ``path``.

    Row order matches ``results`` order, which in turn matches the
    deterministic ``(category, patient_id)`` iteration order built by
    ``_build_working_set`` — satisfying R3's row-order acceptance
    criterion without any additional sort. LF line terminators and no
    quoting beyond the csv module's default minimal quoting keep the
    output byte-identical across runs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(_INDEX_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(_build_index_row(result))


# ──────────────────────────────────────────────────────────────────────────
# Failures sidecar (R2 "failures.txt")
# ──────────────────────────────────────────────────────────────────────────


def _format_failure_line(result: RenderResult) -> str:
    """Format one line of ``failures.txt`` for a failed case.

    R2 pins the format to ``{prefix}_{patient_id:04d} — {render_notes}``
    with an em-dash (not a hyphen). The fallback note protects against
    the degenerate case where ``render_notes`` is somehow empty — no
    index row should ever have an empty note for a failed render, but a
    silent "unknown failure" is safer than an orphan identifier.
    """

    prefix = _category_prefix(result.case.key.category)
    patient_id = result.case.key.patient_id
    notes = result.status.get("render_notes") or "unknown failure"
    return f"{prefix}_{patient_id:04d} — {notes}"


def _write_failures_txt(results: list[RenderResult], path: Path) -> int:
    """Write ``failures.txt`` when at least one case is ``failed``.

    Per R2: the file is NOT created on a clean run. In addition, any
    stale sidecar from a previous run is removed when the current run
    has zero failures — leaving a pre-existing file behind would break
    R4's deterministic DVC output hash (the case_summaries directory is
    tracked as a single ``outs`` entry, so a lingering file mutates the
    hash without any corresponding input change).

    Returns the number of failed cases written.
    """

    failed = [
        r for r in results if r.status["render_status"] == "failed"
    ]
    if not failed:
        path.unlink(missing_ok=True)
        return 0
    with path.open("w") as f:
        for result in failed:
            f.write(_format_failure_line(result))
            f.write("\n")
    return len(failed)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Entry point. Returns the process exit code.

    Exit codes:
        0 — success (per-case failures are recorded but never escalate)
        1 — missing input CSVs or raw dataset directory
        2 — uncaught programming error outside the per-case loop
    """

    for required in (KNOWN_ISSUES_CSV, CLEAN_INDEX_CSV, RAW_DATASET_DIR):
        if not required.exists():
            print(
                f"ERROR: missing required input "
                f"{required.relative_to(REPO_ROOT)}",
                file=sys.stderr,
            )
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    known_issues = _load_known_issues(KNOWN_ISSUES_CSV)
    clean_index = _load_clean_index(CLEAN_INDEX_CSV)
    working_set = _build_working_set(known_issues)

    print(
        f"build_case_summaries: {len(working_set)} working-set cases "
        f"({len(known_issues)} flagged cases total)"
    )

    results: list[RenderResult] = []
    for case in working_set:
        try:
            result = _render_one_case(
                case, clean_index.get(case.key), RAW_DATASET_DIR, OUTPUT_DIR,
            )
        except Exception as exc:
            # Per-case skip-and-log: a single broken case never escapes
            # the loop. T-106 will additionally collect the note for the
            # failures.txt sidecar. Full traceback goes to stderr so a
            # genuinely unexpected crash can be diagnosed from DVC logs
            # without having to re-run the failed case interactively.
            print(
                f"  {case.key.category[0]}_{case.key.patient_id:04d}: "
                f"failed ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            result = RenderResult(
                case=case,
                status=_failed_status(
                    f"uncaught: {type(exc).__name__}: {exc}"
                ),
                png_path=None,
            )
        results.append(result)

    _write_index_csv(results, INDEX_CSV)
    _write_failures_txt(results, FAILURES_TXT)

    n_ok = sum(1 for r in results if r.status["render_status"] == "ok")
    n_partial = sum(1 for r in results if r.status["render_status"] == "partial")
    n_failed = sum(1 for r in results if r.status["render_status"] == "failed")
    print(f"Rendered {n_ok} ok / {n_partial} partial / {n_failed} failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Catastrophic failure outside the per-case loop — print the
        # traceback so DVC logs have something to inspect, then exit
        # non-zero so the user notices a programming error rather than
        # a silent skip.
        traceback.print_exc()
        sys.exit(2)
