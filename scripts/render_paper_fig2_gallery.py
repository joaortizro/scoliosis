"""Render Fig 2 of the paper — gallery of 4 representative cases.

Selection rule: median multiclass Dice within each Cobb severity bucket on
the sealed test set (n=25). Predictions use the 5-fold RB-UNet ensemble
with hflip TTA, matching the headline sentinel ``phase1_2_5fold_TEST.json``.

Output is gitignored per project policy on patient images.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ai.evaluation.seg_metrics import macro_dice_per_image
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.preprocessing.transforms import deterministic_vertical_crop
from ai.training.dataset import preprocess_case

REPO = Path(__file__).resolve().parent.parent
TEST_CSV = REPO / "data/processed/audit_v2_corrected/test_holdout.csv"
# Use the D2 5-fold (RB-UNet trained on IBIO-SD + ERS-18). D2 is the final
# model of the project; its sealed-test result is the headline.
SENTINEL = REPO / "experiments/results/phase1_2_d2_5fold.json"
# Partial-FOV variant (M1a) 5-fold — RB-UNet trained with RandomVerticalCrop
# augmentation. Used for the 5th column (partial-coverage demonstration).
SENTINEL_PF = REPO / "experiments/results/partial_fov_gentle_5fold.json"
# Case to use for the partial-coverage demo (one of the cases already shown
# at full coverage in cols 1–4) and crop parameters.
PARTIAL_CASE_ID = "Scoliosis_196"
PARTIAL_F = 0.5
PARTIAL_MODE = "top"
OUT = Path("/tmp/scoliosis_viz/paper_fig2_gallery.png")


def cobb_bucket(c: float) -> str:
    if pd.isna(c):
        return "Normal"
    if c < 45:
        return "Moderado"
    if c < 65:
        return "Severo"
    return "Muy severo"


BUCKETS = ["Normal", "Moderado", "Severo", "Muy severo"]


def vertebra_palette() -> np.ndarray:
    """17 distinguishable colors for T1..L5 (RGB in [0,1])."""
    cmap = plt.cm.get_cmap("nipy_spectral")
    return np.array([cmap(0.05 + 0.9 * i / 16)[:3] for i in range(17)])


def overlay_mask(ax, image: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.55) -> None:
    ax.imshow(image, cmap="gray")
    rgba = np.zeros((*mask.shape, 4))
    for k in range(1, 18):
        rgba[mask == k, :3] = palette[k - 1]
        rgba[mask == k, 3] = alpha
    ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])


DIFF_COLORS = {
    "correct": (0.10, 0.78, 0.20),   # verde — clase correcta
    "id_err":  (1.00, 0.85, 0.10),   # amarillo — error de ID
    "fn":      (0.10, 0.40, 1.00),   # azul — falso negativo
    "fp":      (0.95, 0.10, 0.10),   # rojo — falso positivo
}


def diff_overlay(ax, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.70) -> None:
    ax.imshow(image, cmap="gray")
    rgba = np.zeros((*gt.shape, 4))
    correct = (gt == pred) & (gt > 0)
    id_err = (gt > 0) & (pred > 0) & (gt != pred)
    fn = (gt > 0) & (pred == 0)
    fp = (gt == 0) & (pred > 0)
    for mask, color in (
        (correct, DIFF_COLORS["correct"]),
        (id_err,  DIFF_COLORS["id_err"]),
        (fn,      DIFF_COLORS["fn"]),
        (fp,      DIFF_COLORS["fp"]),
    ):
        rgba[mask, :3] = color
        rgba[mask, 3] = alpha
    ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    sentinel = json.loads(SENTINEL.read_text())
    fold_dirs = [REPO / f["run_dir"] for f in sentinel["folds"]]
    device = torch.device("cpu")
    predictors = [Predictor(d, device=device) for d in fold_dirs]
    print(f"loaded {len(predictors)} fold predictors")

    p0 = predictors[0]
    test_df = pd.read_csv(TEST_CSV)
    print(f"test cases: {len(test_df)}")

    records: list[dict] = []
    for _, row in test_df.iterrows():
        case = preprocess_case(row, clahe_mode=p0.clahe_mode, roi_crop_mode=p0.roi_crop_mode)
        image = case["image"]  # (1, H, W)
        seg_gt = case["seg"]   # (H, W)

        probs_sum: torch.Tensor | None = None
        for p in predictors:
            probs = p.predict_logits(image, tta="hflip").cpu()
            probs_sum = probs if probs_sum is None else probs_sum + probs
        assert probs_sum is not None
        pred = (probs_sum / len(predictors)).argmax(dim=0)

        pred_oh = torch.zeros((1, NUM_SEG_CLASSES, *pred.shape))
        pred_oh.scatter_(1, pred.long().unsqueeze(0).unsqueeze(0), 1.0)
        target = seg_gt.long().unsqueeze(0)
        dice = macro_dice_per_image(pred_oh, target, num_classes=NUM_SEG_CLASSES).item()

        cobb = row.get("cobb_angle_deg")
        records.append({
            "case_id": row["case_id"],
            "cobb": float(cobb) if not pd.isna(cobb) else float("nan"),
            "bucket": cobb_bucket(cobb),
            "dice": dice,
            "image": image[0].numpy(),
            "seg_gt": seg_gt.numpy(),
            "pred": pred.numpy(),
        })
        cobb_str = f"{cobb:6.1f}" if not pd.isna(cobb) else "  ----"
        print(f"  {row['case_id']:>15s}  cobb={cobb_str}  dice={dice:.4f}")

    # Median Dice per bucket
    selected: dict[str, dict] = {}
    for b in BUCKETS:
        in_b = [r for r in records if r["bucket"] == b]
        if not in_b:
            print(f"WARNING: bucket {b} has 0 cases — skipped")
            continue
        in_b_sorted = sorted(in_b, key=lambda r: r["dice"])
        median_idx = len(in_b_sorted) // 2  # even-length: upper median
        selected[b] = in_b_sorted[median_idx]
        print(f"  bucket {b!r}: n={len(in_b)}  median-Dice case={in_b_sorted[median_idx]['case_id']}  Dice={in_b_sorted[median_idx]['dice']:.4f}")

    # Partial-coverage demonstration: load M1a predictors + run inference on
    # one of the existing cases with a deterministic vertical crop applied.
    sentinel_pf = json.loads(SENTINEL_PF.read_text())
    pf_dirs = [REPO / f["run_dir"] for f in sentinel_pf["folds"]]
    pf_predictors = [Predictor(d, device=device) for d in pf_dirs]
    print(f"loaded {len(pf_predictors)} partial-FOV (M1a) fold predictors")

    # Locate the case for partial-coverage demo
    rec_pc_full = next(r for r in records if r["case_id"] == PARTIAL_CASE_ID)
    # Reconstruct image+seg tensors at preprocessing resolution
    case_row = test_df[test_df["case_id"] == PARTIAL_CASE_ID].iloc[0]
    case_pre = preprocess_case(case_row, clahe_mode=p0.clahe_mode, roi_crop_mode=p0.roi_crop_mode)
    img_t = case_pre["image"]
    seg_t = case_pre["seg"]
    img_crop, seg_crop = deterministic_vertical_crop(img_t, seg_t, f=PARTIAL_F, mode=PARTIAL_MODE)

    probs_sum_pc: torch.Tensor | None = None
    for p in pf_predictors:
        probs = p.predict_logits(img_crop, tta="hflip").cpu()
        probs_sum_pc = probs if probs_sum_pc is None else probs_sum_pc + probs
    assert probs_sum_pc is not None
    pred_pc = (probs_sum_pc / len(pf_predictors)).argmax(dim=0)

    # Dice multiclase only over the visible (cropped) GT region
    pred_oh = torch.zeros((1, NUM_SEG_CLASSES, *pred_pc.shape))
    pred_oh.scatter_(1, pred_pc.long().unsqueeze(0).unsqueeze(0), 1.0)
    target = seg_crop.long().unsqueeze(0)
    dice_pc = macro_dice_per_image(pred_oh, target, num_classes=NUM_SEG_CLASSES).item()
    print(f"  partial-FOV demo: {PARTIAL_CASE_ID} crop f={PARTIAL_F} mode={PARTIAL_MODE}  dice_mc={dice_pc:.4f}")

    partial = {
        "case_id": PARTIAL_CASE_ID,
        "cobb": rec_pc_full["cobb"],
        "f": PARTIAL_F,
        "mode": PARTIAL_MODE,
        "dice": dice_pc,
        "image": img_crop[0].numpy(),
        "seg_gt": seg_crop.numpy(),
        "pred": pred_pc.numpy(),
    }

    buckets_present = [b for b in BUCKETS if b in selected]
    n_cols = len(buckets_present) + 1  # extra column for partial-coverage demo
    palette = vertebra_palette()

    fig, axes = plt.subplots(4, n_cols, figsize=(2.8 * n_cols, 10.5), gridspec_kw={"hspace": 0.10, "wspace": 0.04})

    for col, b in enumerate(buckets_present):
        r = selected[b]
        cobb_str = f"Cobb GT: {r['cobb']:.1f}°" if not np.isnan(r["cobb"]) else "Cobb GT: —"

        ax = axes[0, col]
        ax.imshow(r["image"], cmap="gray")
        ax.set_title(f"{r['case_id']}  ·  {cobb_str}\nDice mc: {r['dice']:.3f}", fontsize=9.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

        overlay_mask(axes[1, col], r["image"], r["seg_gt"], palette)
        overlay_mask(axes[2, col], r["image"], r["pred"], palette)
        diff_overlay(axes[3, col], r["image"], r["seg_gt"], r["pred"])

    # 5th column — partial-coverage demo
    pc_col = len(buckets_present)
    cobb_pc = f"Cobb GT: {partial['cobb']:.1f}°" if not np.isnan(partial["cobb"]) else "Cobb GT: —"
    axes[0, pc_col].imshow(partial["image"], cmap="gray")
    axes[0, pc_col].set_title(
        f"{partial['case_id']}  ·  crop $f={partial['f']}$ ({partial['mode']})\n"
        f"Dice mc: {partial['dice']:.3f}   (variante partial-FOV)",
        fontsize=9.5, pad=4,
    )
    axes[0, pc_col].set_xticks([]); axes[0, pc_col].set_yticks([])
    overlay_mask(axes[1, pc_col], partial["image"], partial["seg_gt"], palette)
    overlay_mask(axes[2, pc_col], partial["image"], partial["pred"], palette)
    diff_overlay(axes[3, pc_col], partial["image"], partial["seg_gt"], partial["pred"])

    # Bucket headers above each column
    headers = list(buckets_present) + ["Cobertura parcial"]
    for col, b in enumerate(headers):
        x_center = 0.07 + (col + 0.5) * (0.99 - 0.07) / n_cols
        fig.text(x_center, 0.965, b, ha="center", va="bottom", fontsize=13, weight="bold")

    # Row labels via dedicated text on figure (figure-coord y-centers)
    row_labels = ["Radiografía", "Etiquetas\nGT", "Predicción\nfinal (D2)", "Diferencia"]
    n_rows = 4
    top, bottom = 0.93, 0.07
    row_height = (top - bottom) / n_rows
    row_y_centers = [top - (i + 0.5) * row_height for i in range(n_rows)]
    for row_idx, (lbl, yc) in enumerate(zip(row_labels, row_y_centers)):
        fig.text(0.015, yc, lbl, ha="left", va="center", fontsize=11, weight="bold", rotation=90)

    # Legend for the diff row (bottom of figure)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=DIFF_COLORS["correct"], edgecolor="none", label="Clase correcta (GT = pred)"),
        Patch(facecolor=DIFF_COLORS["id_err"],  edgecolor="none", label="Error de ID (ambos fg, distinta clase)"),
        Patch(facecolor=DIFF_COLORS["fn"],      edgecolor="none", label="Falso negativo (GT fg, pred bg)"),
        Patch(facecolor=DIFF_COLORS["fp"],      edgecolor="none", label="Falso positivo (GT bg, pred fg)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.99)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
