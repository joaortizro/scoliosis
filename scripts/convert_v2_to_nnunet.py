"""Convert Scoliosis_v2_corrected to nnU-Net v2 dataset format.

Output layout:
    <nnUNet_raw>/Dataset001_Spine/
        imagesTr/case_<sid>_0000.nii.gz   (1, H, W) grayscale (channel 0)
        labelsTr/case_<sid>.nii.gz        (1, H, W) uint8 0..17
        dataset.json                      18 classes, 1 channel

Source: ``data/processed/audit_v2_corrected/clean_index.csv`` filtered to
``trainable_rows`` (matches the trainer's split materialization).

Image preprocessing matches the trainer:
    * resize to 512 x 256 BILINEAR (image)
    * resize to 512 x 256 NEAREST  (mask)
    * remap multiclass IDs 1..17 -> seg classes 1..17 (0 = bg)

We deliberately do NOT apply roi_crop here: nnU-Net runs its own
preprocessing pipeline (resampling, normalization) and we want it to see
the full-image cases. The Phase 1.x roi_crop=from_mask gain (~+0.02 Dice)
should be subsumed by nnU-Net's heavy augmentation budget.

The sealed test_holdout (25 cases) is excluded — only trainable rows are
written, so the 5-fold CV nnU-Net runs internally on this subset will
match the comparison perimeter we used for Phase 1.2 5-fold.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from ai.preprocessing.segmentation import (
    NUM_SEG_CLASSES,
    TARGET_VERTEBRA_IDS,
    remap_to_target_classes,
)
from ai.training.dataset import read_gray, read_mask, resize_image, resize_mask
from ai.training.splits import trainable_rows

import pandas as pd

log = logging.getLogger(__name__)

IMG_H = 512
IMG_W = 256


def _to_nifti(arr_2d: np.ndarray) -> nib.Nifti1Image:
    """Wrap a (H, W) array as a (H, W, 1) NIfTI with identity affine."""
    if arr_2d.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr_2d.shape}")
    arr = arr_2d[:, :, np.newaxis]  # (H, W, 1) — Z axis = 1
    return nib.Nifti1Image(arr, affine=np.eye(4))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-index", default="data/processed/audit_v2_corrected/clean_index.csv")
    parser.add_argument("--out-dir", required=True, help="$nnUNet_raw")
    parser.add_argument("--dataset-id", default="001")
    parser.add_argument("--dataset-name", default="Spine")
    parser.add_argument("--min-target-count", type=int, default=14)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(args.clean_index)
    rows = trainable_rows(df, min_target_count=args.min_target_count)
    log.info("trainable rows: %d (full clean_index: %d)", len(rows), len(df))

    base = Path(args.out_dir) / f"Dataset{args.dataset_id}_{args.dataset_name}"
    images_tr = base / "imagesTr"
    labels_tr = base / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    log.info("writing to %s", base)

    n_written = 0
    for _, row in rows.iterrows():
        category = str(row["category"])  # "Scoliosis" or "Normal"
        patient_id = int(row["patient_id"])
        # case_id mirrors the splits.materialize key — disambiguate by category
        # because Scoliosis 1 and Normal 1 are different patients.
        case_id = f"{category[0].lower()}_{patient_id:03d}"  # e.g. s_001, n_023

        image_raw = read_gray(Path(row["image_path"]))
        mask_raw = read_mask(Path(row["multiclass_mask_path"]))
        image_np = resize_image(image_raw, IMG_H, IMG_W)
        mask_np = resize_mask(mask_raw, IMG_H, IMG_W)
        seg_np = remap_to_target_classes(mask_np, target_ids=TARGET_VERTEBRA_IDS)

        # nnU-Net expects: image as float32 (it normalizes internally),
        # label as uint8 (or any int up to int32) with values 0..C-1.
        nib.save(_to_nifti(image_np.astype(np.float32)), images_tr / f"{case_id}_0000.nii.gz")
        nib.save(_to_nifti(seg_np.astype(np.uint8)), labels_tr / f"{case_id}.nii.gz")
        n_written += 1
        if n_written % 50 == 0:
            log.info("  wrote %d cases", n_written)

    log.info("wrote %d cases", n_written)

    # dataset.json — nnU-Net v2 schema.
    label_dict = {"background": 0}
    for i, vid in enumerate(TARGET_VERTEBRA_IDS, start=1):
        label_dict[f"vertebra_{vid}"] = i  # vertebra_1..vertebra_17 → classes 1..17
    dataset_json = {
        "channel_names": {"0": "X-ray"},
        "labels": label_dict,
        "numTraining": n_written,
        "file_ending": ".nii.gz",
        "name": f"Dataset{args.dataset_id}_{args.dataset_name}",
        "description": (
            "MaIA Scoliosis v2-corrected, 17-class T1..L5 vertebra segmentation "
            "from 2D AP X-rays. Resized to 512x256 BILINEAR (image) / NEAREST (mask). "
            "Trainable rows only (status in {ok, warn} AND target_vertebrae_count >= 14). "
            "Sealed test holdout (25 cases) NOT included."
        ),
    }
    (base / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    log.info("wrote dataset.json with %d classes", NUM_SEG_CLASSES)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
