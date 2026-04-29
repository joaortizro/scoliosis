"""Build Scoliosis_Dataset_v2_corrected from v2 + colleague's corrected ID masks.

A colleague hand-painted 147 multiclass masks (palette-PNG workflow) so missing
T1..L5 vertebrae would be present. This script mirrors the v2 dataset into a
new ``Scoliosis_Dataset_v2_corrected/`` root, overlays the corrected ID PNGs,
and regenerates every artifact derived from the multiclass mask:

    - LabelBinaryJPG/      (255 where ID > 0, else 0)
    - LabelMultiClass_Gray_JPG/   (per-ID grayscale palette from the dict)
    - LabelMultiClass_Color_JPG/  (per-ID RGB palette from the dict)

Files copied as-is (no derivation):
    - indice_dataset.csv, resumen_*.csv, reporte_*.csv, dictionaries

Files symlinked (large, unchanged):
    - Scoliosis/, Normal/                  (radiographs)
    - RadiographMetrics/                   (Cobb metrics — focus is segmentation
                                            first, Cobb regen is a separate step)

Usage
-----
    python scripts/build_corrected_dataset.py \\
        --src data/raw/Scoliosis_Dataset_v2 \\
        --corrections /home/ortiz/vertebra_mask_correction/outputs/LabelMultiClass_ID_PNG_corrected \\
        --dst data/raw/Scoliosis_Dataset_v2_corrected
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT: Path = Path(__file__).resolve().parent.parent

ID_MASK_DIR = "LabelMultiClass_ID_PNG"
BINARY_MASK_DIR = "LabelBinaryJPG"
GRAY_MASK_DIR = "LabelMultiClass_Gray_JPG"
COLOR_MASK_DIR = "LabelMultiClass_Color_JPG"

SYMLINK_TARGETS = ("Scoliosis", "Normal", "RadiographMetrics")
COPY_FILES = (
    "indice_dataset.csv",
    "resumen_version_final_T1_T12_L1_L5.csv",
    "reporte_por_mascara_version_final.csv",
    "resumen_diccionario_original_35_entidades.csv",
    "diccionario_etiquetas_T1_T12_L1_L5.json",
    "README.md",
)


def load_palettes(dict_path: Path) -> tuple[dict[int, int], dict[int, tuple[int, int, int]]]:
    """Read grayscale + RGB palette tables from the v2 dictionary JSON."""
    data = json.loads(dict_path.read_text(encoding="utf-8"))
    gray = {int(k): int(v) for k, v in data["mascara_multiclase_grises_jpg"].items()}
    color = {int(k): tuple(int(c) for c in v) for k, v in data["mascara_multiclase_color_rgb"].items()}
    return gray, color


def id_mask_to_array(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def write_binary_mask(id_arr: np.ndarray, out: Path) -> None:
    binary = np.where(id_arr > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(out, format="JPEG", quality=95)


def write_gray_viz(id_arr: np.ndarray, gray_palette: dict[int, int], out: Path) -> None:
    lut = np.zeros(256, dtype=np.uint8)
    for class_id, gray_value in gray_palette.items():
        lut[class_id] = gray_value
    gray = lut[id_arr]
    Image.fromarray(gray, mode="L").save(out, format="JPEG", quality=95)


def write_color_viz(
    id_arr: np.ndarray, rgb_palette: dict[int, tuple[int, int, int]], out: Path
) -> None:
    h, w = id_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in rgb_palette.items():
        rgb[id_arr == class_id] = color
    Image.fromarray(rgb, mode="RGB").save(out, format="JPEG", quality=95)


def link_or_copy(src: Path, dst: Path, prefer_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if prefer_symlink:
        dst.symlink_to(src.resolve())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def stage_static(src_root: Path, dst_root: Path) -> None:
    """Symlink large unchanged dirs + copy small CSV/JSON metadata."""
    for name in SYMLINK_TARGETS:
        src = src_root / name
        if not src.exists():
            print(f"[skip] missing {src}", file=sys.stderr)
            continue
        link_or_copy(src, dst_root / name, prefer_symlink=True)

    for name in COPY_FILES:
        src = src_root / name
        if not src.exists():
            print(f"[skip] missing {src}", file=sys.stderr)
            continue
        link_or_copy(src, dst_root / name, prefer_symlink=False)


def build_masks(
    src_root: Path,
    dst_root: Path,
    corrections_dir: Path,
    gray_palette: dict[int, int],
    rgb_palette: dict[int, tuple[int, int, int]],
) -> dict[str, int]:
    """Overlay corrected ID masks, then regen binary + gray + color from final IDs."""
    src_id_dir = src_root / ID_MASK_DIR
    dst_id_dir = dst_root / ID_MASK_DIR
    dst_bin_dir = dst_root / BINARY_MASK_DIR
    dst_gray_dir = dst_root / GRAY_MASK_DIR
    dst_color_dir = dst_root / COLOR_MASK_DIR

    for d in (dst_id_dir, dst_bin_dir, dst_gray_dir, dst_color_dir):
        d.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_id_dir.glob("LabelMulti_*.png"))
    if not src_files:
        raise RuntimeError(f"No multiclass PNGs found in {src_id_dir}")

    counts = {"total": 0, "corrected": 0, "from_v2": 0, "shape_mismatch": 0}

    for src_id_path in tqdm(src_files, desc="masks"):
        base = src_id_path.stem  # LabelMulti_S_21
        corrected_path = corrections_dir / f"{base}.png"

        if corrected_path.exists():
            src_arr = id_mask_to_array(src_id_path)
            cor_arr = id_mask_to_array(corrected_path)
            if cor_arr.shape != src_arr.shape:
                counts["shape_mismatch"] += 1
                print(
                    f"[shape mismatch] {base}: v2 {src_arr.shape} vs corrected {cor_arr.shape} — "
                    "falling back to v2",
                    file=sys.stderr,
                )
                final_arr = src_arr
            else:
                final_arr = cor_arr
                counts["corrected"] += 1
        else:
            final_arr = id_mask_to_array(src_id_path)
            counts["from_v2"] += 1

        Image.fromarray(final_arr, mode="L").save(
            dst_id_dir / f"{base}.png", format="PNG", compress_level=6
        )

        prefix, group, num = base.split("_", 2)  # LabelMulti, S/N, id
        binary_name = f"Label_{group}_{num}.jpg"
        viz_name = f"{base}.jpg"

        write_binary_mask(final_arr, dst_bin_dir / binary_name)
        write_gray_viz(final_arr, gray_palette, dst_gray_dir / viz_name)
        write_color_viz(final_arr, rgb_palette, dst_color_dir / viz_name)

        counts["total"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src",
        type=Path,
        default=REPO_ROOT / "data/raw/Scoliosis_Dataset_v2",
        help="Source v2 dataset root",
    )
    p.add_argument(
        "--corrections",
        type=Path,
        default=Path("/home/ortiz/vertebra_mask_correction/outputs/LabelMultiClass_ID_PNG_corrected"),
        help="Directory of colleague's corrected LabelMulti_*.png files",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=REPO_ROOT / "data/raw/Scoliosis_Dataset_v2_corrected",
        help="Destination root for the corrected dataset",
    )
    p.add_argument("--force", action="store_true", help="Wipe --dst before building")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    corrections: Path = args.corrections.resolve()

    if not src.exists():
        print(f"[error] source not found: {src}", file=sys.stderr)
        return 1
    if not corrections.exists():
        print(f"[error] corrections dir not found: {corrections}", file=sys.stderr)
        return 1

    if dst.exists() and args.force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"src         : {src}")
    print(f"corrections : {corrections}")
    print(f"dst         : {dst}")

    gray_palette, rgb_palette = load_palettes(src / "diccionario_etiquetas_T1_T12_L1_L5.json")

    stage_static(src, dst)
    counts = build_masks(src, dst, corrections, gray_palette, rgb_palette)

    print("\nDone.")
    print(f"  total masks      : {counts['total']}")
    print(f"  used corrected   : {counts['corrected']}")
    print(f"  kept v2 original : {counts['from_v2']}")
    print(f"  shape mismatches : {counts['shape_mismatch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
