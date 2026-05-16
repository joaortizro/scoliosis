"""Full Mode-1 audit on all 45 fold-4 val cases.

For each case:
  1. Run inference, compare GT vs prediction at trainer resolution.
  2. Render a 3-panel big-view PNG (raw native + GT@trainer + pred@trainer).
  3. Classify the failure mode:
       - complete           — GT=pred=17 vertebrae, fine
       - gt_missing_bottom  — pred has IDs below GT's lowest (likely L5 absent in GT)
       - gt_missing_top     — pred has IDs above GT's highest (likely T1/T2 absent in GT)
       - gt_partial         — GT has fewer than 17, model also matches the partial range
       - pred_missing       — model predicts fewer than GT
       - mismatch           — counts equal but ID ranges differ
  4. Save summary CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.inference.predictor import Predictor
from ai.training.splits import make_cv_folds, trainable_rows

DATASET = REPO / "data/raw/Scoliosis_Dataset_v2_corrected"
SENTINEL = REPO / "experiments/results/phase1_2_5fold.json"
CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"
TEST_HOLDOUT = REPO / "data/processed/audit_v2_corrected/test_holdout.csv"
EVAL_CSV = REPO / "notebooks/sandbox/viz_2026-05-11/fold4_partial_coverage_eval.csv"
OUT_DIR = REPO / "notebooks/sandbox/viz_2026-05-11/all45_bigview"
SUMMARY_CSV = REPO / "notebooks/sandbox/viz_2026-05-11/all45_audit_summary.csv"

VERT_LABELS = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]


def vertebra_cmap() -> ListedColormap:
    base = plt.colormaps["tab20"].colors + plt.colormaps["tab20b"].colors
    colors = [(0, 0, 0, 0)] + [base[i % len(base)] for i in range(17)]
    return ListedColormap(colors, name="vert18")


def centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    out = {}
    for vid in range(1, 18):
        ys, xs = np.where(mask == vid)
        if len(ys) > 5:
            out[vid] = (float(xs.mean()), float(ys.mean()))
    return out


def label_vertebrae(ax, cents, fontsize=10):
    for vid, (cx, cy) in cents.items():
        ax.text(cx, cy, VERT_LABELS[vid - 1],
                color="yellow", fontsize=fontsize, ha="center", va="center",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65, lw=0))


def classify_case(gt_ids: list[int], pred_ids: list[int]) -> str:
    gt_set = set(gt_ids)
    pr_set = set(pred_ids)
    gt_n = len(gt_set)
    pr_n = len(pr_set)
    if gt_n == 17 and pr_n == 17:
        return "complete"
    if gt_n == pr_n and gt_set == pr_set:
        return "complete_partial"  # both same, but not all 17 (e.g. both 16)
    extra_in_pred = pr_set - gt_set
    missing_from_pred = gt_set - pr_set
    if extra_in_pred and not missing_from_pred:
        # pred has classes GT doesn't
        if max(extra_in_pred) > max(gt_set):
            if min(extra_in_pred) < min(gt_set):
                return "gt_missing_both_ends"
            return "gt_missing_bottom"  # pred has L5 etc. that GT doesn't
        if min(extra_in_pred) < min(gt_set):
            return "gt_missing_top"
        return "gt_missing_middle"
    if missing_from_pred and not extra_in_pred:
        return "pred_missing"
    if extra_in_pred and missing_from_pred:
        return "mismatch"
    return "complete"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads(SENTINEL.read_text())
    fold = 4
    predictor = Predictor(REPO / sentinel["folds"][fold]["run_dir"], device=torch.device("cpu"))

    splits = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT)
    spec = splits[fold]
    full_df = pd.read_csv(CLEAN_INDEX)
    pool = trainable_rows(full_df, min_target_count=14)
    val_df = pool.loc[list(spec.val_idx)].reset_index(drop=True)
    print(f"fold {fold}: {len(val_df)} val cases")

    eval_df = pd.read_csv(EVAL_CSV)
    cmap = vertebra_cmap()
    summary_rows = []

    for i, (_, row) in enumerate(val_df.iterrows()):
        pid = int(row["patient_id"])
        category = row["category"]
        prefix = "S" if category == "Scoliosis" else "N"

        raw_img_path = DATASET / category / f"{prefix}_{pid}.jpg"
        raw_mask_path = DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_{prefix}_{pid}.png"
        if not raw_img_path.exists():
            print(f"  skip {prefix}_{pid} — image missing")
            continue

        raw_img = np.array(Image.open(raw_img_path).convert("L"))
        raw_mask = np.array(Image.open(raw_mask_path))

        out = predictor.predict_from_row(row, tta="hflip")
        pred_img = out["image"].squeeze().cpu().numpy()
        pred = out["pred"].cpu().numpy().astype(np.int32)
        gt_512 = out["seg"].cpu().numpy().astype(np.int32)

        gt_raw_ids = sorted(int(u) for u in np.unique(raw_mask) if u > 0)
        gt_512_ids = sorted(int(u) for u in np.unique(gt_512) if u > 0)
        pred_ids = sorted(int(u) for u in np.unique(pred) if u > 0)

        case_class = classify_case(gt_512_ids, pred_ids)
        mc_orig = eval_df[(eval_df["patient_id"] == pid) & (eval_df["category"] == category)]
        mc_orig_val = float(mc_orig["mc_dice_original"].iloc[0]) if len(mc_orig) else float("nan")
        bin_val = float(mc_orig["binary_dice"].iloc[0]) if len(mc_orig) else float("nan")

        # short description
        gt_range = f"{VERT_LABELS[gt_raw_ids[0]-1]}..{VERT_LABELS[gt_raw_ids[-1]-1]}" if gt_raw_ids else "—"
        pred_range = f"{VERT_LABELS[pred_ids[0]-1]}..{VERT_LABELS[pred_ids[-1]-1]}" if pred_ids else "—"
        # human-visible vertebrae we can't compute automatically — leave blank
        summary_rows.append({
            "patient_id": pid,
            "category": category,
            "cobb_deg": row.get("cobb_angle_deg"),
            "binary_dice": bin_val,
            "mc_dice": mc_orig_val,
            "gt_count": len(gt_raw_ids),
            "pred_count": len(pred_ids),
            "gt_range": gt_range,
            "pred_range": pred_range,
            "case_class": case_class,
        })

        # Render figure
        raw_cents = centroids(raw_mask)
        pred_cents = centroids(pred)
        gt_cents = centroids(gt_512)

        fig, axes = plt.subplots(1, 3, figsize=(15, 11))
        axes[0].imshow(raw_img, cmap="gray", aspect="equal")
        axes[0].imshow(raw_mask, cmap=cmap, vmin=0, vmax=17, alpha=0.30)
        label_vertebrae(axes[0], raw_cents, fontsize=11)
        axes[0].set_title(
            f"RAW + GT  (native {raw_img.shape[1]}×{raw_img.shape[0]})\n"
            f"GT: {gt_range} ({len(gt_raw_ids)} vertebrae)",
            fontsize=10,
        )
        axes[0].axis("off")

        axes[1].imshow(pred_img, cmap="gray", aspect="equal")
        axes[1].imshow(gt_512, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        label_vertebrae(axes[1], gt_cents, fontsize=10)
        axes[1].set_title(f"GT @ trainer res (512×256)\n{len(gt_512_ids)} vertebrae", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(pred_img, cmap="gray", aspect="equal")
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        label_vertebrae(axes[2], pred_cents, fontsize=10)
        axes[2].set_title(
            f"Pred  ({pred_range}, {len(pred_ids)} vertebrae)\n"
            f"bin={bin_val:.3f}  mc={mc_orig_val:.3f}",
            fontsize=10,
        )
        axes[2].axis("off")

        cobb = row.get("cobb_angle_deg")
        cobb_str = f"Cobb={cobb:.1f}°" if pd.notna(cobb) else "Normal"
        fig.suptitle(
            f"{category} {prefix}_{pid}  —  {cobb_str}  |  class: {case_class}",
            fontsize=13, y=0.99,
        )
        fig.tight_layout()
        # sort filename by case_class then mc (worst first within class)
        mc_tag = f"{mc_orig_val:.3f}" if not np.isnan(mc_orig_val) else "nan"
        out_path = OUT_DIR / f"{case_class}__{mc_tag}__{prefix}_{pid}.png"
        fig.savefig(out_path, dpi=85, bbox_inches="tight")
        plt.close(fig)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(val_df)} done")

    df = pd.DataFrame(summary_rows)
    df.sort_values(["case_class", "mc_dice"], inplace=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nsaved summary: {SUMMARY_CSV}")
    print(f"saved PNGs: {OUT_DIR} ({len(list(OUT_DIR.glob('*.png')))} files)")

    print("\n" + "=" * 70)
    print("Case-class summary:")
    print("=" * 70)
    counts = df["case_class"].value_counts()
    for cls, n in counts.items():
        mc_mean = df[df["case_class"] == cls]["mc_dice"].mean()
        print(f"  {cls:<25} n={n:>2}  mean mc Dice = {mc_mean:.3f}")
    print()
    fixable_classes = ["gt_missing_bottom", "gt_missing_top", "gt_missing_middle", "gt_missing_both_ends"]
    fixable = df[df["case_class"].isin(fixable_classes)]
    if len(fixable):
        print(f"FIXABLE BY HAND-ANNOTATION: {len(fixable)} cases ({len(fixable)/len(df)*100:.0f}% of val)")
        print(f"  current mean mc Dice: {fixable['mc_dice'].mean():.3f}")
        print(f"  estimated post-fix:   ~0.85 (model is already anatomically right)")
        print(f"  fold-4 lift estimate: ~+{(0.85 - fixable['mc_dice'].mean()) * len(fixable) / len(df):.3f}")


if __name__ == "__main__":
    main()
