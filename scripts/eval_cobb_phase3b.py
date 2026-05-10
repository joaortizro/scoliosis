"""Phase 3b — Cobb MAE evaluation against Wu/BoostNet GT (corner-based).

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §3.

Loads a YOLOv8-Pose checkpoint, runs inference on a v2 fold's val images,
computes predicted Cobb via the canonical Wu/BoostNet pairwise-max formula
(`cobb_from_detected_vertebrae`), and compares to:

    Primary GT: Wu/BoostNet Cobb computed from v2 GT corners (re-baselined).
    Secondary: MaIA's `angulo_cobb_deg` from clean_index.csv.

Usage:
    python scripts/eval_cobb_phase3b.py \\
        --weights ai/models/checkpoints/yolo_vertebra/<run>/weights/best.pt \\
        --fold 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ai.detection.postprocess import cobb_from_detected_vertebrae
from ai.evaluation.cobb_endplate import cobb_from_keypoints_endplate
from ai.preprocessing.keypoints import multiclass_mask_to_keypoints

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "yolo_pose_datasets"

CONF_THRESHOLD = 0.25
TOPN_DETECTIONS = 20


def _load_clean_index_lookup() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_INDEX)
    df["stem"] = df["category"].str[0] + "_" + df["patient_id"].astype(str)
    return df.set_index("stem")


def _wu_cobb_gt_from_mask(mask_path: Path) -> float:
    mask = np.array(Image.open(mask_path))
    kps = multiclass_mask_to_keypoints(mask)
    cobb, _ = cobb_from_keypoints_endplate(kps)
    return cobb


def _predict_cobb(model, image_path: Path, device: str) -> tuple[float, int]:
    """Run YOLO inference, return (predicted Cobb, n_detections_after_filter)."""
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        max_det=TOPN_DETECTIONS,
        device=device,
        verbose=False,
    )
    r = results[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return 0.0, 0
    detections = r.keypoints.xy.cpu().numpy()  # (N, 4, 2)
    if detections.shape[1] != 4:
        return 0.0, 0
    cobb, info = cobb_from_detected_vertebrae(detections)
    return cobb, info["n_valid_vertebrae"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="path to best.pt")
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="0")
    ap.add_argument("--output-name", default=None,
                    help="sentinel filename suffix (default: derived from weights path)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    fold_val_images = DATASET_ROOT / f"v2_fold_{args.fold}" / "images" / "val"
    val_image_paths = sorted(fold_val_images.glob("*.jpg"))
    if not val_image_paths:
        raise RuntimeError(f"no val images at {fold_val_images}")
    print(f"Evaluating fold {args.fold}: {len(val_image_paths)} val images")

    lookup = _load_clean_index_lookup()

    rows = []
    for img_path in val_image_paths:
        stem = img_path.stem  # e.g. "S_104"
        meta = lookup.loc[stem]
        wu_gt = _wu_cobb_gt_from_mask(Path(meta["multiclass_mask_path"]))
        maia_gt = float(meta["cobb_angle_deg"]) if pd.notna(meta["cobb_angle_deg"]) else float("nan")
        pred, n_det = _predict_cobb(model, img_path, args.device)
        rows.append({
            "stem": stem,
            "category": meta["category"],
            "wu_gt_deg": wu_gt,
            "maia_gt_deg": maia_gt,
            "pred_deg": pred,
            "n_detections": n_det,
            "err_wu_deg": abs(pred - wu_gt) if np.isfinite(wu_gt) else float("nan"),
            "err_maia_deg": abs(pred - maia_gt) if np.isfinite(maia_gt) else float("nan"),
        })

    df_results = pd.DataFrame(rows)

    # Stats: only on cases where GT is finite (Normal cases have no MaIA Cobb;
    # Wu Cobb is computable for all cases since masks exist).
    wu_valid = df_results["err_wu_deg"].notna()
    maia_valid = df_results["err_maia_deg"].notna()

    out: dict[str, object] = {
        "weights": str(args.weights),
        "fold": args.fold,
        "n_val_images": len(df_results),
        "n_wu_valid": int(wu_valid.sum()),
        "n_maia_valid": int(maia_valid.sum()),
        "wu_mae_deg": float(df_results.loc[wu_valid, "err_wu_deg"].mean()),
        "wu_median_err_deg": float(df_results.loc[wu_valid, "err_wu_deg"].median()),
        "wu_max_err_deg": float(df_results.loc[wu_valid, "err_wu_deg"].max()),
        "maia_mae_deg": float(df_results.loc[maia_valid, "err_maia_deg"].mean()),
        "maia_median_err_deg": float(df_results.loc[maia_valid, "err_maia_deg"].median()),
        "maia_max_err_deg": float(df_results.loc[maia_valid, "err_maia_deg"].max()),
        "mean_n_detections": float(df_results["n_detections"].mean()),
    }

    output_name = args.output_name or args.weights.parts[-3]  # run-dir name
    sentinel_path = REPO_ROOT / "experiments" / "results" / f"cobb_eval_{output_name}_fold{args.fold}.json"
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(json.dumps(out, indent=2))

    csv_path = sentinel_path.with_suffix(".csv")
    df_results.to_csv(csv_path, index=False)

    print("\n=== Cobb evaluation ===")
    print(f"n_val={out['n_val_images']}  n_wu_valid={out['n_wu_valid']}  n_maia_valid={out['n_maia_valid']}")
    print(f"Wu/BoostNet GT — MAE={out['wu_mae_deg']:.2f}°  median={out['wu_median_err_deg']:.2f}°  max={out['wu_max_err_deg']:.2f}°")
    print(f"MaIA GT       — MAE={out['maia_mae_deg']:.2f}°  median={out['maia_median_err_deg']:.2f}°  max={out['maia_max_err_deg']:.2f}°")
    print(f"mean detections per image = {out['mean_n_detections']:.1f}")
    print(f"sentinel: {sentinel_path}")
    print(f"per-case csv: {csv_path}")


if __name__ == "__main__":
    main()
