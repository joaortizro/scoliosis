"""ID-mask ↔ RGB-color editor for the v2_corrected_x2 GIMP workflow.

Adapted from Jorge Oñate's `mask_correction_workflow.py` (shared via Google
Drive). Same API (`inspect_one`, `export_editable_color_mask`,
`convert_edited_color_to_id`, `show_bad_colors_red`, …) so the workflow doc
applies verbatim. Only the path layout is re-pointed at our repo:

  input  ID masks      → data/raw/Scoliosis_Dataset_v2_corrected/LabelMultiClass_ID_PNG/
  input  radiographs   → data/raw/Scoliosis_Dataset_v2_corrected/{Normal,Scoliosis}/
                          (resolved automatically by the N_ vs S_ prefix)
  output corrected     → data/raw/Scoliosis_Dataset_v2_corrected_x2/LabelMultiClass_ID_PNG/
  scratch editable RGB → .local/mask_correction_x2/color_edit/
  scratch logs         → .local/mask_correction_x2/logs/

Usage:
    python -i scripts/mask_correction_x2/mask_correction_workflow.py
    >>> inspect_one("LabelMulti_S_156")
    >>> export_editable_color_mask("LabelMulti_S_156")
    # edit in GIMP, save back to the same path …
    >>> convert_edited_color_to_id("LabelMulti_S_156")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Input dataset (v2_corrected — Jorge's Phase-1 output, our current training set).
    SOURCE_DATASET_DIR: Path = REPO_ROOT / "data" / "raw" / "Scoliosis_Dataset_v2_corrected"
    RADIOGRAPH_ROOT: Path = SOURCE_DATASET_DIR
    ID_MASK_DIR: Path = SOURCE_DATASET_DIR / "LabelMultiClass_ID_PNG"
    REPORT_CSV: Path = SOURCE_DATASET_DIR / "reporte_por_mascara_version_final.csv"

    # Output dataset (v2_corrected_x2 — DVC-tracked, only the new corrections).
    OUTPUT_DATASET_DIR: Path = REPO_ROOT / "data" / "raw" / "Scoliosis_Dataset_v2_corrected_x2"
    CORRECTED_ID_DIR: Path = OUTPUT_DATASET_DIR / "LabelMultiClass_ID_PNG"

    # Gitignored scratch space (editable RGB + per-case logs).
    SCRATCH_ROOT: Path = REPO_ROOT / ".local" / "mask_correction_x2"
    COLOR_EDIT_DIR: Path = SCRATCH_ROOT / "color_edit"
    LOG_DIR: Path = SCRATCH_ROOT / "logs"
    CORRECTION_LOG_CSV: Path = LOG_DIR / "correction_log.csv"

    FILE_EXTENSION: str = ".png"
    SUSPICIOUS_NONZERO_ID_THRESHOLD: int = 5


CFG = Config()


# ============================================================
# PALETTE / LABEL DEFINITIONS
# ============================================================

ID_TO_NAME: Dict[int, str] = {
    0: "background",
    1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6",
    7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11", 12: "T12",
    13: "L1", 14: "L2", 15: "L3", 16: "L4", 17: "L5",
}

ID_TO_HEX: Dict[int, str] = {
    0: "#000000",
    1: "#F2D10C",  2: "#EBF20C",  3: "#C4F20C",  4: "#9CF20C",
    5: "#75F20C",  6: "#4DF20C",  7: "#26F20C",  8: "#0CF219",
    9: "#0CF240",  10: "#0CF268", 11: "#0CF28F", 12: "#0CF2B7",
    13: "#0CF2DE", 14: "#0CDEF2", 15: "#0CB7F2", 16: "#0C8FF2",
    17: "#0C68F2",
}

ALLOWED_IDS = set(ID_TO_NAME.keys())


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


ID_TO_RGB: Dict[int, Tuple[int, int, int]] = {
    class_id: hex_to_rgb(hex_code) for class_id, hex_code in ID_TO_HEX.items()
}

RGB_TO_ID: Dict[Tuple[int, int, int], int] = {
    rgb: class_id for class_id, rgb in ID_TO_RGB.items()
}


# ============================================================
# SETUP HELPERS
# ============================================================

def _assert_safe_output_path(path: Path) -> None:
    """Hard rule: never write into the source dataset dir. Anywhere else is fine."""
    abs_path = path.resolve()
    abs_source = CFG.SOURCE_DATASET_DIR.resolve()
    try:
        abs_path.relative_to(abs_source)
    except ValueError:
        return  # path is outside source dataset — safe
    raise PermissionError(
        f"Refusing to write inside the source dataset dir.\n"
        f"  attempted: {abs_path}\n"
        f"  source:    {abs_source}\n"
        f"  rule: originals are read-only; corrected output must land in "
        f"{CFG.OUTPUT_DATASET_DIR} or {CFG.SCRATCH_ROOT}."
    )


def ensure_directories() -> None:
    _assert_safe_output_path(CFG.COLOR_EDIT_DIR)
    _assert_safe_output_path(CFG.CORRECTED_ID_DIR)
    _assert_safe_output_path(CFG.LOG_DIR)
    CFG.COLOR_EDIT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.CORRECTED_ID_DIR.mkdir(parents=True, exist_ok=True)
    CFG.LOG_DIR.mkdir(parents=True, exist_ok=True)


def print_setup_summary() -> None:
    print("Repo root:           ", REPO_ROOT)
    print("Source dataset:      ", CFG.SOURCE_DATASET_DIR)
    print("Original ID masks:   ", CFG.ID_MASK_DIR)
    print("Report CSV:          ", CFG.REPORT_CSV)
    print("Output dataset (x2): ", CFG.OUTPUT_DATASET_DIR)
    print("Corrected ID masks:  ", CFG.CORRECTED_ID_DIR)
    print("Editable color RGB:  ", CFG.COLOR_EDIT_DIR)
    print("Correction log:      ", CFG.CORRECTION_LOG_CSV)


# ============================================================
# FILE RESOLUTION HELPERS
# ============================================================

def strip_mask_prefix_for_radiograph(base_name: str) -> str:
    """LabelMulti_S_156 -> S_156 / LabelMulti_N_67 -> N_67 / pass-through otherwise."""
    prefix = "LabelMulti_"
    if base_name.startswith(prefix):
        return base_name[len(prefix):]
    return base_name


def radiograph_subdir_for(base_name: str) -> Path:
    """Pick Normal/ vs Scoliosis/ subdir based on the N_/S_ prefix in base_name."""
    radiograph_base = strip_mask_prefix_for_radiograph(base_name)
    if radiograph_base.startswith("N_"):
        return CFG.RADIOGRAPH_ROOT / "Normal"
    if radiograph_base.startswith("S_"):
        return CFG.RADIOGRAPH_ROOT / "Scoliosis"
    raise ValueError(
        f"Cannot determine radiograph subdir for '{base_name}' — "
        f"expected stem starting with N_ or S_, got '{radiograph_base}'"
    )


def resolve_input_path_flexible(
    base_dir: Path,
    base_name: str,
    exts: Optional[List[str]] = None,
) -> Path:
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for ext in exts:
        candidate = base_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No file found for base name '{base_name}' in {base_dir} "
        f"with extensions: {exts}"
    )


def radiograph_path(base_name: str) -> Path:
    radiograph_base = strip_mask_prefix_for_radiograph(base_name)
    return resolve_input_path_flexible(
        radiograph_subdir_for(base_name),
        radiograph_base,
        exts=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],
    )


def id_mask_path(base_name: str) -> Path:
    return resolve_input_path_flexible(CFG.ID_MASK_DIR, base_name, exts=[".png"])


def editable_color_mask_path(base_name: str) -> Path:
    return CFG.COLOR_EDIT_DIR / f"{base_name}_COLOR_EDIT.png"


def corrected_id_mask_path(base_name: str) -> Path:
    return CFG.CORRECTED_ID_DIR / f"{base_name}.png"


# ============================================================
# IMAGE LOADERS
# ============================================================

def load_radiograph(base_name: str) -> Image.Image:
    return Image.open(radiograph_path(base_name)).copy()


def load_id_mask(base_name: str) -> Image.Image:
    return Image.open(id_mask_path(base_name)).copy()


def id_mask_to_array(mask_img: Image.Image) -> np.ndarray:
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


# ============================================================
# MASK ANALYSIS / VISUALIZATION
# ============================================================

def get_unique_ids(mask_array: np.ndarray) -> List[int]:
    return sorted(np.unique(mask_array).tolist())


def validate_allowed_ids(mask_array: np.ndarray) -> Tuple[bool, List[int]]:
    unique_ids = get_unique_ids(mask_array)
    unexpected = [x for x in unique_ids if x not in ALLOWED_IDS]
    return (len(unexpected) == 0, unexpected)


def create_bright_visualization(mask_array: np.ndarray) -> np.ndarray:
    h, w = mask_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in ID_TO_RGB.items():
        rgb[mask_array == class_id] = color
    return rgb


def inspect_one(base_name: str, show_plot: bool = True) -> Dict:
    rad_img = load_radiograph(base_name)
    mask_img = load_id_mask(base_name)

    rad_arr = np.array(rad_img)
    mask_arr = id_mask_to_array(mask_img)

    if rad_arr.shape[:2] != mask_arr.shape[:2]:
        raise ValueError(
            f"Size mismatch for {base_name}: "
            f"radiograph shape={rad_arr.shape[:2]}, mask shape={mask_arr.shape[:2]}"
        )

    unique_ids = get_unique_ids(mask_arr)
    nonzero_ids = [x for x in unique_ids if x != 0]
    is_valid, unexpected_ids = validate_allowed_ids(mask_arr)
    suspicious = len(nonzero_ids) < CFG.SUSPICIOUS_NONZERO_ID_THRESHOLD

    print("=" * 70)
    print(f"Filename: {base_name}")
    print(f"Radiograph: {radiograph_path(base_name)}")
    print(f"Mask: {id_mask_path(base_name)}")
    print(f"Image size: {mask_arr.shape[1]} x {mask_arr.shape[0]}")
    print(f"Unique IDs present: {unique_ids}")
    print(f"Nonzero IDs present: {nonzero_ids}")
    print(f"Allowed ID validation: {is_valid}")
    if not is_valid:
        print(f"Unexpected IDs found: {unexpected_ids}")
    print(f"Suspicious by threshold (< {CFG.SUSPICIOUS_NONZERO_ID_THRESHOLD} nonzero IDs): {suspicious}")

    bright_rgb = create_bright_visualization(mask_arr)

    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        if rad_arr.ndim == 2:
            axes[0].imshow(rad_arr, cmap="gray")
        else:
            axes[0].imshow(rad_arr)
        axes[0].set_title(f"Radiograph\n{base_name}")
        axes[0].axis("off")

        axes[1].imshow(mask_arr, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title("Original ID Mask Visualization")
        axes[1].axis("off")

        axes[2].imshow(bright_rgb, interpolation="nearest")
        axes[2].set_title("Bright Fixed-Palette Visualization")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "filename": base_name,
        "unique_ids": unique_ids,
        "nonzero_ids": nonzero_ids,
        "is_valid": is_valid,
        "unexpected_ids": unexpected_ids,
        "suspicious": suspicious,
    }


# ============================================================
# EXPORT EDITABLE COLOR MASK
# ============================================================

def export_editable_color_mask(base_name: str, overwrite: bool = False) -> Path:
    mask_img = load_id_mask(base_name)
    mask_arr = id_mask_to_array(mask_img)

    is_valid, unexpected_ids = validate_allowed_ids(mask_arr)
    if not is_valid:
        raise ValueError(
            f"Cannot export editable color mask for {base_name}. "
            f"Unexpected IDs in original mask: {unexpected_ids}"
        )

    out_path = editable_color_mask_path(base_name)
    _assert_safe_output_path(out_path)
    if out_path.exists() and not overwrite:
        print(f"Editable color mask already exists: {out_path}")
        return out_path

    rgb_arr = create_bright_visualization(mask_arr)
    Image.fromarray(rgb_arr, mode="RGB").save(out_path, format="PNG", compress_level=0)
    print(f"Saved editable color mask: {out_path}")
    return out_path


def batch_export_editable_color_masks(base_names: List[str], overwrite: bool = False) -> List[Path]:
    exported = []
    for base_name in base_names:
        try:
            exported.append(export_editable_color_mask(base_name, overwrite=overwrite))
        except Exception as e:
            print(f"[FAILED] {base_name}: {e}")
    return exported


# ============================================================
# CONVERT EDITED COLOR PNG BACK TO ID PNG
# ============================================================

def load_edited_color_mask(edited_color_path: Path) -> np.ndarray:
    if not edited_color_path.exists():
        raise FileNotFoundError(f"Edited color mask not found: {edited_color_path}")
    return np.array(Image.open(edited_color_path).convert("RGB"), dtype=np.uint8)


def find_unexpected_colors(rgb_arr: np.ndarray) -> List[Tuple[int, int, int]]:
    flat = rgb_arr.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    unexpected = []
    for color in unique_colors:
        color_tuple = tuple(int(x) for x in color.tolist())
        if color_tuple not in RGB_TO_ID:
            unexpected.append(color_tuple)
    return unexpected


def convert_edited_color_to_id(
    base_name: str,
    edited_color_path: Optional[Path] = None,
    save_log: bool = True,
) -> Path:
    original_mask_img = load_id_mask(base_name)
    original_mask_arr = id_mask_to_array(original_mask_img)
    original_unique_ids = get_unique_ids(original_mask_arr)

    if edited_color_path is None:
        edited_color_path = editable_color_mask_path(base_name)

    edited_rgb_arr = load_edited_color_mask(edited_color_path)

    if edited_rgb_arr.shape[:2] != original_mask_arr.shape[:2]:
        raise ValueError(
            f"Edited color mask size mismatch for {base_name}: "
            f"edited={edited_rgb_arr.shape[:2]}, original={original_mask_arr.shape[:2]}"
        )

    unexpected_colors = find_unexpected_colors(edited_rgb_arr)
    if unexpected_colors:
        readable = [f"RGB{c}" for c in unexpected_colors]
        raise ValueError(
            f"Unexpected colors found in edited mask for {base_name}: {readable}\n"
            "This usually means anti-aliasing, soft brush edges, opacity changes, or "
            "a color outside the fixed palette was used."
        )

    h, w, _ = edited_rgb_arr.shape
    corrected_id_arr = np.zeros((h, w), dtype=np.uint8)
    for rgb, class_id in RGB_TO_ID.items():
        matches = np.all(edited_rgb_arr == np.array(rgb, dtype=np.uint8), axis=-1)
        corrected_id_arr[matches] = class_id

    is_valid, unexpected_ids = validate_allowed_ids(corrected_id_arr)
    if not is_valid:
        raise ValueError(
            f"Converted ID mask for {base_name} contains invalid IDs: {unexpected_ids}"
        )

    out_path = corrected_id_mask_path(base_name)
    _assert_safe_output_path(out_path)
    Image.fromarray(corrected_id_arr, mode="L").save(out_path, format="PNG", compress_level=0)

    corrected_unique_ids = get_unique_ids(corrected_id_arr)
    correction_applied = (
        original_unique_ids != corrected_unique_ids
        or not np.array_equal(original_mask_arr, corrected_id_arr)
    )

    if save_log:
        append_correction_log(
            filename=base_name,
            original_unique_ids=original_unique_ids,
            corrected_unique_ids=corrected_unique_ids,
            correction_applied=correction_applied,
        )

    print(f"Saved corrected ID PNG: {out_path}")
    print(f"Original unique IDs:  {original_unique_ids}")
    print(f"Corrected unique IDs: {corrected_unique_ids}")
    print(f"Correction applied:   {correction_applied}")
    return out_path


# ============================================================
# BAD-COLOR DEBUG (per the doc's troubleshooting recipe)
# ============================================================

def show_bad_colors_red(base_name: str) -> None:
    """Highlight non-palette pixels in red. Save preview to LOG_DIR + display."""
    rgb = load_edited_color_mask(editable_color_mask_path(base_name))
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    bad_mask = np.zeros(rgb.shape[:2], dtype=bool)
    print("Unexpected colors:")
    for color, count in zip(colors, counts):
        color_tuple = tuple(int(x) for x in color.tolist())
        if color_tuple not in RGB_TO_ID:
            print(f"  RGB{color_tuple} -> {int(count)} pixels")
            bad_mask |= np.all(rgb == np.array(color_tuple, dtype=np.uint8), axis=-1)

    preview = rgb.copy()
    preview[bad_mask] = np.array([255, 0, 0], dtype=np.uint8)

    CFG.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CFG.LOG_DIR / f"{base_name}_BAD_COLORS_RED.png"
    Image.fromarray(preview).save(out_path, format="PNG", compress_level=0)
    print(f"\nSaved preview with bad pixels in red:\n  {out_path}")

    plt.figure(figsize=(8, 14))
    plt.imshow(preview, interpolation="nearest")
    plt.title(f"Bad colors highlighted in red: {base_name}")
    plt.axis("off")
    plt.show()


# ============================================================
# LOGGING
# ============================================================

def append_correction_log(
    filename: str,
    original_unique_ids: List[int],
    corrected_unique_ids: List[int],
    correction_applied: bool,
) -> None:
    ensure_directories()
    row = {
        "filename": filename,
        "original_unique_ids": " ".join(map(str, original_unique_ids)),
        "corrected_unique_ids": " ".join(map(str, corrected_unique_ids)),
        "correction_applied": correction_applied,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    file_exists = CFG.CORRECTION_LOG_CSV.exists()
    with open(CFG.CORRECTION_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# REPORT CSV SUPPORT (kept for compatibility; we drive batches from
# our own triage CSV in practice — see triage_audit_cases.py)
# ============================================================

def load_report_csv() -> pd.DataFrame:
    if not CFG.REPORT_CSV.exists():
        raise FileNotFoundError(f"Report CSV not found: {CFG.REPORT_CSV}")
    df = pd.read_csv(CFG.REPORT_CSV)
    print(f"Loaded report CSV: {CFG.REPORT_CSV}  ({len(df)} rows, cols={list(df.columns)})")
    return df


def guess_filename_column(df: pd.DataFrame) -> str:
    for cand in ["filename", "image_name", "image", "basename", "file", "name", "stem"]:
        for col in df.columns:
            if str(col).strip().lower() == cand:
                return col
    return df.columns[0]


def shortlist_from_report(max_items: int = 20) -> List[str]:
    df = load_report_csv()
    filename_col = guess_filename_column(df)
    names = [Path(name).stem for name in df[filename_col].astype(str).tolist()][:max_items]
    print(f"Using filename column: {filename_col} | shortlist size: {len(names)}")
    return names


def export_shortlist_from_report(max_items: int = 20, overwrite: bool = False) -> List[Path]:
    return batch_export_editable_color_masks(shortlist_from_report(max_items=max_items), overwrite=overwrite)


def inspect_shortlist_from_report(max_items: int = 5) -> None:
    for name in shortlist_from_report(max_items=max_items):
        try:
            inspect_one(name, show_plot=True)
        except Exception as e:
            print(f"[FAILED INSPECT] {name}: {e}")


def validate_corrected_mask_file(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Corrected mask not found: {path}")
    arr = id_mask_to_array(Image.open(path))
    valid, unexpected = validate_allowed_ids(arr)
    unique_ids = get_unique_ids(arr)
    print(f"Validated: {path}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Allowed IDs only: {valid}")
    if not valid:
        print(f"Unexpected IDs: {unexpected}")
    return {
        "path": str(path),
        "unique_ids": unique_ids,
        "valid": valid,
        "unexpected_ids": unexpected,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_directories()
    print_setup_summary()
    print("\nReady.")
    print("Examples:")
    print('  inspect_one("LabelMulti_S_156")')
    print('  export_editable_color_mask("LabelMulti_S_156")')
    print('  convert_edited_color_to_id("LabelMulti_S_156")')
    print('  show_bad_colors_red("LabelMulti_S_156")')
