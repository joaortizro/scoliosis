"""Phase 3b — robustified Cobb evaluation.

Same as `eval_cobb_phase3b.py` but evaluates THREE Cobb formula variants
on the same predictions:

    naive       : max(slope) - min(slope)            — current spec
    pctile_5_95 : percentile-trim 5/95 then max-min  — outlier suppression
    segment     : segment-aware (S-curve safe)        — advisor-recommended

Both predicted and GT-corner slopes go through the same formula per row,
so each variant is a self-consistent comparison.

Usage:
    python scripts/eval_cobb_phase3b_robust.py --weights <best.pt> --fold 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ai.detection.postprocess import order_centroids_by_chain
from ai.evaluation.cobb_endplate import (
    cobb_from_midline_slopes,
    cobb_pctile_trim,
    cobb_segment_aware,
    compute_midline_slopes,
)
from ai.preprocessing.keypoints import multiclass_mask_to_keypoints

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "yolo_pose_datasets"

CONF_THRESHOLD = 0.25
TOPN_DETECTIONS = 20

VARIANTS = ("naive", "pctile_5_95", "segment")


def _load_clean_index_lookup() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_INDEX)
    df["stem"] = df["category"].str[0] + "_" + df["patient_id"].astype(str)
    return df.set_index("stem")


def _slopes_from_corners(corners: np.ndarray) -> np.ndarray:
    """corners shape (N, 4, 2) → (N,) midline slopes (deg). NaN where invalid."""
    if corners.size == 0:
        return np.array([], dtype=np.float64)
    n = len(corners)
    slopes = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        c = corners[i]
        if not np.isfinite(c).all():
            continue
        top_mid = 0.5 * (c[0] + c[1])
        bot_mid = 0.5 * (c[2] + c[3])
        dx = bot_mid[0] - top_mid[0]
        dy = bot_mid[1] - top_mid[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        slopes[i] = float(np.degrees(np.arctan2(dx, dy)))
    return slopes


def _ordered_slopes_from_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners by chain, return ordered slopes (1-D)."""
    if len(corners) == 0:
        return np.array([], dtype=np.float64)
    centroids = corners.mean(axis=1)
    order = order_centroids_by_chain(centroids)
    return _slopes_from_corners(corners[order])


def _cobb_variant(slopes_1d: np.ndarray, variant: str) -> float:
    """Pad to length-17 NaN array if needed, then run the variant formula.

    Naive uses cobb_from_midline_slopes which expects (17,). Segment + pctile
    work on the variable-length valid subset directly.
    """
    if len(slopes_1d) == 0:
        return 0.0
    if variant == "naive":
        padded = np.full(max(17, len(slopes_1d)), np.nan, dtype=np.float64)
        padded[: len(slopes_1d)] = slopes_1d
        # Trim to 17 if longer (unusual)
        if len(padded) > 17:
            valid = np.isfinite(padded)
            valid_slopes = padded[valid]
            if len(valid_slopes) < 4:
                return 0.0
            return float(valid_slopes.max() - valid_slopes.min())
        cobb, _ = cobb_from_midline_slopes(padded)
        return cobb
    if variant == "pctile_5_95":
        cobb, _ = cobb_pctile_trim(slopes_1d, 5.0, 95.0)
        return cobb
    if variant == "segment":
        cobb, _ = cobb_segment_aware(slopes_1d)
        return cobb
    raise ValueError(f"unknown variant: {variant}")


def _gt_slopes_chain(mask_path: Path) -> np.ndarray:
    """GT corners (17×4) → 17-vertebra slopes in chain order (NaN for missing)."""
    mask = np.array(Image.open(mask_path))
    kps = multiclass_mask_to_keypoints(mask)  # (68, 2)
    return compute_midline_slopes(kps)  # (17,)


def _pred_slopes_chain(model, image_path: Path, device: str) -> np.ndarray:
    """Run YOLO inference → (N,) ordered midline slopes."""
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        max_det=TOPN_DETECTIONS,
        device=device,
        verbose=False,
    )
    r = results[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return np.array([], dtype=np.float64)
    detections = r.keypoints.xy.cpu().numpy()  # (N, 4, 2)
    if detections.shape[1] != 4:
        return np.array([], dtype=np.float64)
    return _ordered_slopes_from_corners(detections)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="0")
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    fold_val_images = DATASET_ROOT / f"v2_fold_{args.fold}" / "images" / "val"
    val_image_paths = sorted(fold_val_images.glob("*.jpg"))
    if not val_image_paths:
        raise RuntimeError(f"no val images at {fold_val_images}")
    print(f"Evaluating fold {args.fold}: {len(val_image_paths)} val images, variants={VARIANTS}")

    lookup = _load_clean_index_lookup()

    rows: list[dict] = []
    for img_path in val_image_paths:
        stem = img_path.stem
        meta = lookup.loc[stem]
        gt_slopes = _gt_slopes_chain(Path(meta["multiclass_mask_path"]))
        pred_slopes = _pred_slopes_chain(model, img_path, args.device)
        maia_gt = float(meta["cobb_angle_deg"]) if pd.notna(meta["cobb_angle_deg"]) else float("nan")

        row: dict[str, object] = {
            "stem": stem,
            "category": meta["category"],
            "maia_gt_deg": maia_gt,
            "n_gt_slopes": int(np.isfinite(gt_slopes).sum()),
            "n_pred_slopes": int(np.isfinite(pred_slopes).sum()),
        }
        for v in VARIANTS:
            gt_cobb = _cobb_variant(gt_slopes[np.isfinite(gt_slopes)], v)
            pred_cobb = _cobb_variant(pred_slopes[np.isfinite(pred_slopes)], v)
            row[f"gt_{v}_deg"] = gt_cobb
            row[f"pred_{v}_deg"] = pred_cobb
            row[f"err_vs_gt_{v}"] = abs(pred_cobb - gt_cobb)
            if np.isfinite(maia_gt):
                row[f"err_vs_maia_{v}"] = abs(pred_cobb - maia_gt)
            else:
                row[f"err_vs_maia_{v}"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    summary: dict[str, object] = {
        "weights": str(args.weights),
        "fold": args.fold,
        "n_val_images": len(df),
        "n_maia_valid": int(df["maia_gt_deg"].notna().sum()),
        "variants": list(VARIANTS),
        "per_variant": {},
    }
    for v in VARIANTS:
        valid_gt = df[f"err_vs_gt_{v}"].notna()
        valid_maia = df[f"err_vs_maia_{v}"].notna()
        summary["per_variant"][v] = {
            "vs_gt_self": {
                "mae_deg": float(df.loc[valid_gt, f"err_vs_gt_{v}"].mean()),
                "median_deg": float(df.loc[valid_gt, f"err_vs_gt_{v}"].median()),
                "max_deg": float(df.loc[valid_gt, f"err_vs_gt_{v}"].max()),
            },
            "vs_maia": {
                "mae_deg": float(df.loc[valid_maia, f"err_vs_maia_{v}"].mean()),
                "median_deg": float(df.loc[valid_maia, f"err_vs_maia_{v}"].median()),
                "max_deg": float(df.loc[valid_maia, f"err_vs_maia_{v}"].max()),
            },
        }

    output_name = args.output_name or args.weights.parts[-3]
    sentinel_path = REPO_ROOT / "experiments" / "results" / f"cobb_eval_robust_{output_name}_fold{args.fold}.json"
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(json.dumps(summary, indent=2))
    df.to_csv(sentinel_path.with_suffix(".csv"), index=False)

    print("\n=== Cobb evaluation (robust variants) ===")
    print(f"n_val={summary['n_val_images']}  n_maia_valid={summary['n_maia_valid']}")
    print(f"{'variant':<14}{'vs gt-self':<35}{'vs MaIA':<35}")
    for v in VARIANTS:
        s = summary["per_variant"][v]
        gtself = f"MAE={s['vs_gt_self']['mae_deg']:.2f} med={s['vs_gt_self']['median_deg']:.2f} max={s['vs_gt_self']['max_deg']:.2f}"
        maia = f"MAE={s['vs_maia']['mae_deg']:.2f} med={s['vs_maia']['median_deg']:.2f} max={s['vs_maia']['max_deg']:.2f}"
        print(f"{v:<14}{gtself:<35}{maia:<35}")
    print(f"sentinel: {sentinel_path}")


if __name__ == "__main__":
    main()
