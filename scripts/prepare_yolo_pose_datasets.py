"""Materialize YOLOv8-Pose-compatible dataset directories.

Phase 3b.2b. Idempotent — re-running overwrites label files.

Output structure:
    data/processed/yolo_pose_datasets/
    ├── roboflow_pretrain/
    │   ├── images/{train,valid,test}/*.jpg            (symlinks into data/raw/roboflow_scoliosis_v16/images/...)
    │   ├── labels/{train,valid,test}/*.txt            (YOLO-Pose labels w/ dummy kpts visibility=0)
    │   └── data.yaml
    └── v2_fold_{0..4}/
        ├── images/{train,val}/*.jpg                   (symlinks into data/raw/Scoliosis_Dataset_v2_corrected/...)
        ├── labels/{train,val}/*.txt                   (YOLO-Pose labels w/ corner kpts visibility=2)
        └── data.yaml

Roboflow filter: only images with ≥14 vertebrae are kept for pretrain
(partial-coverage cases are reserved for Q-22 OOD eval).

v2 fold splits: same as Phase 1.2 5-fold (KFold seed=42 on trainable
clean_index rows via ai.training.splits.make_cv_folds).

Sealed test holdout (25 cases) is excluded from all training splits.
Verified by tests/test_no_leakage.py.

Usage:
    python scripts/prepare_yolo_pose_datasets.py [--roboflow-only | --v2-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from ai.detection.data_conversion import (
    format_yolo_pose_line,
    multiclass_mask_to_yolo_pose,
    roboflow_bbox_line_to_yolo_pose,
)
from ai.detection.roboflow_filter import filter_roboflow_split

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_SRC = REPO_ROOT / "data" / "raw" / "roboflow_scoliosis_v16"
V2_CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
V2_SEALED_TEST = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "test_holdout.csv"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "yolo_pose_datasets"

MIN_VERTEBRAE_ROBOFLOW = 14
N_FOLDS = 5
SPLIT_SEED = 42


def _safe_symlink(src: Path, dst: Path) -> None:
    """Symlink src → dst, replacing any existing dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def prepare_roboflow_pretrain() -> dict:
    """Build the Roboflow pretrain dataset with bbox-only labels (kpt loss masked)."""
    out_dir = OUT_ROOT / "roboflow_pretrain"
    stats: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        labels_src = ROBOFLOW_SRC / "labels" / split
        images_src = ROBOFLOW_SRC / "images" / split
        kept_stems = filter_roboflow_split(labels_src, min_vertebrae=MIN_VERTEBRAE_ROBOFLOW)
        # For valid/test, optionally keep all (so eval set isn't filtered by training rules)
        # but we follow spec §5.1 — apply filter uniformly across pretrain set.
        labels_out = out_dir / "labels" / split
        images_out = out_dir / "images" / split
        labels_out.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)

        n_kept_bboxes = 0
        for stem in kept_stems:
            src_label = labels_src / f"{stem}.txt"
            lines: list[str] = []
            for line in src_label.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                lbl = roboflow_bbox_line_to_yolo_pose(line)
                if lbl is not None:
                    lines.append(format_yolo_pose_line(lbl))
                    n_kept_bboxes += 1
            (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            src_image = images_src / f"{stem}.jpg"
            if src_image.exists():
                _safe_symlink(src_image, images_out / src_image.name)
        stats[f"{split}_images"] = len(kept_stems)
        stats[f"{split}_bboxes"] = n_kept_bboxes
        print(f"  {split}: {len(kept_stems)} images, {n_kept_bboxes} vertebra bboxes")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "vertebra"},
        "nc": 1,
        "kpt_shape": [4, 3],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return stats


def _build_5fold_splits(trainable_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """KFold(n=5, shuffle=True, seed=42) on trainable indices."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    indices = np.arange(len(trainable_df))
    return list(kf.split(indices))


def prepare_v2_folds() -> dict:
    """Build per-fold YOLO-Pose datasets from v2 multiclass masks."""
    df = pd.read_csv(V2_CLEAN_INDEX)
    sealed_ids = set(pd.read_csv(V2_SEALED_TEST)["patient_id"].astype(str).tolist())

    # Trainable: status in {ok, warn} AND target_vertebrae_count >= 14 AND not in sealed test
    trainable = df[
        (df["status"].isin(["ok", "warn"]))
        & (df["target_vertebrae_count"] >= 14)
        & (~df["patient_id"].astype(str).isin(sealed_ids))
    ].reset_index(drop=True)
    assert len(trainable) > 0, "no trainable v2 cases after sealed-test exclusion"

    splits = _build_5fold_splits(trainable)
    stats: dict[str, int] = {"n_trainable": len(trainable), "n_sealed_test": len(sealed_ids)}
    print(f"v2: {len(trainable)} trainable cases ({len(sealed_ids)} sealed-test excluded)")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_dir = OUT_ROOT / f"v2_fold_{fold_idx}"
        stats[f"fold_{fold_idx}_train"] = len(train_idx)
        stats[f"fold_{fold_idx}_val"] = len(val_idx)
        print(f"  fold {fold_idx}: {len(train_idx)} train / {len(val_idx)} val")

        for split_name, indices in [("train", train_idx), ("val", val_idx)]:
            labels_out = fold_dir / "labels" / split_name
            images_out = fold_dir / "images" / split_name
            labels_out.mkdir(parents=True, exist_ok=True)
            images_out.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                row = trainable.iloc[idx]
                mask = np.array(Image.open(row["multiclass_mask_path"]))
                labels = multiclass_mask_to_yolo_pose(mask)
                stem = f"{row['category'][0]}_{int(row['patient_id'])}"
                lines = [format_yolo_pose_line(lbl) for lbl in labels]
                (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                _safe_symlink(Path(row["image_path"]), images_out / f"{stem}.jpg")

        data_yaml = {
            "path": str(fold_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "vertebra"},
            "nc": 1,
            "kpt_shape": [4, 3],
        }
        (fold_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow-only", action="store_true")
    ap.add_argument("--v2-only", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, dict] = {}

    if not args.v2_only:
        print("=== Roboflow pretrain ===")
        all_stats["roboflow"] = prepare_roboflow_pretrain()
    if not args.roboflow_only:
        print("=== v2 5-fold splits ===")
        all_stats["v2"] = prepare_v2_folds()

    stats_path = OUT_ROOT / "preparation_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nStats written to {stats_path}")


if __name__ == "__main__":
    main()
