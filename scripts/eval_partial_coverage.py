"""Partial-coverage-aware multi-class Dice on fold 4 val (n=45).

Hypothesis from the GT audit: the catastrophic mc-Dice on the 9 worst
cases comes from two distinct failure modes:
  - Mode 1: GT has < 17 vertebrae (L5 cropped out of field), but model
    predicts 17 → ID cascade kills per-class IoU even though anatomy is correct.
  - Mode 2: GT has 17, model has 17, centroids align within a few px,
    but per-vertebra boundary IoU is killed by small-object precision.

Fix proposed for Mode 1: only score per-class Dice over IDs that appear
in the GT mask. The "phantom" classes the model predicts that don't
exist in GT are excluded (they're not wrong anatomically; they're
counted by the original eval as 0 IoU and drag the per-class mean down).

This script:
  1. Runs fold 4 inference on all 45 val cases.
  2. Computes three metrics per case:
       - binary Dice (any-fg vs any-fg)
       - mc Dice original (mean over all 17 classes, NaN if absent in BOTH gt & pred)
       - mc Dice partial-coverage-aware (mean over GT-present classes only)
  3. Classifies each case as Mode 1 / Mode 2 / clean based on GT coverage
     and centroid alignment.
  4. Prints summary stats + saves CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.inference.predictor import Predictor
from ai.training.splits import make_cv_folds, trainable_rows

CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"
TEST_HOLDOUT = REPO / "data/processed/audit_v2_corrected/test_holdout.csv"
SENTINEL = REPO / "experiments/results/phase1_2_5fold.json"


def per_class_dice(pred: np.ndarray, gt: np.ndarray, n_classes: int = 18) -> list[float]:
    """Per-class Dice; NaN when neither pred nor gt has the class."""
    out: list[float] = []
    for c in range(1, n_classes):
        p = pred == c
        g = gt == c
        denom = p.sum() + g.sum()
        if denom == 0:
            out.append(float("nan"))
        else:
            out.append(float(2 * (p & g).sum() / denom))
    return out


def per_class_dice_partial_coverage(pred: np.ndarray, gt: np.ndarray, n_classes: int = 18) -> list[float]:
    """Per-class Dice scored ONLY over classes present in the GT mask.

    Classes absent in GT (but possibly predicted) return NaN — they are
    excluded from the per-case mean. Classes present in GT score normally
    against the corresponding pred class (zero if model didn't predict it).
    """
    out: list[float] = []
    gt_classes = set(int(u) for u in np.unique(gt) if u > 0)
    for c in range(1, n_classes):
        if c not in gt_classes:
            out.append(float("nan"))  # exclude — not in GT field of view
            continue
        p = pred == c
        g = gt == c
        denom = p.sum() + g.sum()
        out.append(float(2 * (p & g).sum() / denom) if denom else 0.0)
    return out


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_fg = pred > 0
    gt_fg = gt > 0
    denom = pred_fg.sum() + gt_fg.sum()
    return float(2 * (pred_fg & gt_fg).sum() / denom) if denom else 1.0


def centroid_y(mask: np.ndarray, vid: int) -> float | None:
    ys, _ = np.where(mask == vid)
    return float(ys.mean()) if len(ys) > 5 else None


def classify_mode(gt: np.ndarray, pred: np.ndarray) -> tuple[str, dict]:
    """Mode 1 (partial coverage cascade) vs Mode 2 (boundary precision) vs clean."""
    gt_ids = set(int(u) for u in np.unique(gt) if u > 0)
    pr_ids = set(int(u) for u in np.unique(pred) if u > 0)
    gt_n = len(gt_ids)
    pr_n = len(pr_ids)
    # mean abs y-centroid offset for shared IDs
    shared = sorted(gt_ids & pr_ids)
    offsets = []
    for vid in shared:
        g = centroid_y(gt, vid)
        p = centroid_y(pred, vid)
        if g is not None and p is not None:
            offsets.append(abs(p - g))
    mean_offset = float(np.mean(offsets)) if offsets else float("nan")
    info = {"gt_count": gt_n, "pred_count": pr_n, "mean_centroid_dy": mean_offset}
    if gt_n < 17 and pr_n > gt_n:
        return "mode1_partial_coverage", info
    if gt_n == 17 and pr_n == 17 and mean_offset < 10:
        return "mode2_boundary_precision", info
    return "other", info


def main() -> None:
    sentinel = json.loads(SENTINEL.read_text())
    fold = 4
    run_dir = REPO / sentinel["folds"][fold]["run_dir"]
    print(f"loading fold {fold} from {run_dir.name}")
    predictor = Predictor(run_dir, device=torch.device("cpu"))

    splits = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT)
    spec = splits[fold]
    full_df = pd.read_csv(CLEAN_INDEX)
    pool = trainable_rows(full_df, min_target_count=14)
    val_df = pool.loc[list(spec.val_idx)].reset_index(drop=True)
    print(f"fold {fold} val cases: {len(val_df)}")

    rows = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        out = predictor.predict_from_row(row, tta="hflip")
        gt = out["seg"].cpu().numpy().astype(np.int32)
        pr = out["pred"].cpu().numpy().astype(np.int32)
        d_bin = binary_dice(pr, gt)
        d_mc_orig = float(np.nanmean(per_class_dice(pr, gt)))
        d_mc_partial = float(np.nanmean(per_class_dice_partial_coverage(pr, gt)))
        mode, info = classify_mode(gt, pr)
        rows.append({
            "patient_id": int(row["patient_id"]),
            "category": row["category"],
            "cobb_deg": row.get("cobb_angle_deg"),
            "gt_count": info["gt_count"],
            "pred_count": info["pred_count"],
            "mean_centroid_dy": round(info["mean_centroid_dy"], 1) if not np.isnan(info["mean_centroid_dy"]) else float("nan"),
            "mode": mode,
            "binary_dice": d_bin,
            "mc_dice_original": d_mc_orig,
            "mc_dice_partial_coverage": d_mc_partial,
            "lift": d_mc_partial - d_mc_orig,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(val_df)} done")

    df = pd.DataFrame(rows).sort_values("mc_dice_original")
    out_csv = REPO / "notebooks/sandbox/viz_2026-05-11/fold4_partial_coverage_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    bin_m = df["binary_dice"].mean()
    mc_orig_m = df["mc_dice_original"].mean()
    mc_pc_m = df["mc_dice_partial_coverage"].mean()
    reported = sentinel["folds"][fold]["best_val_dice"]

    print("\n" + "=" * 70)
    print(f"FOLD {fold} — n={len(df)} val cases")
    print("=" * 70)
    print(f"reported (trainer best_val_dice):  {reported:.4f}")
    print(f"binary Dice (any fg):              {bin_m:.4f}")
    print(f"mc Dice (original eval):           {mc_orig_m:.4f}")
    print(f"mc Dice (partial-coverage-aware):  {mc_pc_m:.4f}   Δ = +{mc_pc_m - mc_orig_m:+.4f}")

    print("\nMode breakdown:")
    mode_summary = df.groupby("mode").agg(
        n=("patient_id", "count"),
        mean_mc_orig=("mc_dice_original", "mean"),
        mean_mc_pc=("mc_dice_partial_coverage", "mean"),
        mean_lift=("lift", "mean"),
    ).round(3)
    print(mode_summary.to_string())

    print("\n9 worst by original mc (now re-scored):")
    worst = df.head(9)[["patient_id", "gt_count", "pred_count", "mode",
                        "mc_dice_original", "mc_dice_partial_coverage", "lift"]]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
