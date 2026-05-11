"""Rigorous anatomical verification of Mode-1 cases.

For each case classified `gt_missing_bottom` (the strongest claim — that
v2 is missing L5 on cases where L5 IS anatomically visible):
  1. Render a ZOOMED crop of the bottom 35% of the radiograph at native resolution.
  2. Overlay GT vs model prediction in distinct colors.
  3. Mark the position of GT's lowest labeled vertebra and the model's
     lowest predicted vertebra.
  4. Save large enough for visual landmark inspection (iliac crest / sacrum).

Expected anatomical landmarks at the bottom of a spinal X-ray:
  - L5 vertebral body: above the iliac crest line, last clear quadrilateral
  - L5/S1 junction: where the spine angles into the sacrum
  - S1 (sacrum): triangular bone below L5, fused, no clear vertebral body shape
  - Iliac crests: curve outward to the sides at the level of L4-L5

If the model's "L5" prediction sits on a clear vertebral-body shape
ABOVE the iliac crest → GT missed L5 (claim holds).
If it sits on the sacrum (no vertebral body shape) → model hallucinating.
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

DATASET = REPO / "data/raw/Scoliosis_Dataset_v2_corrected"
SENTINEL = REPO / "experiments/results/phase1_2_5fold.json"
CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"
SUMMARY = REPO / "notebooks/sandbox/viz_2026-05-11/all45_audit_summary.csv"
OUT_DIR = REPO / "notebooks/sandbox/viz_2026-05-11/anatomy_verify"

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


def upscale_pred_to_native(pred_512x256: np.ndarray, native_h: int, native_w: int) -> np.ndarray:
    """Upscale trainer-resolution prediction back to native image size via NEAREST."""
    img = Image.fromarray(pred_512x256.astype(np.uint8))
    img_native = img.resize((native_w, native_h), Image.NEAREST)
    return np.array(img_native).astype(np.int32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    sentinel = json.loads(SENTINEL.read_text())
    predictor = Predictor(REPO / sentinel["folds"][4]["run_dir"], device=torch.device("cpu"))
    clean = pd.read_csv(CLEAN_INDEX)
    cmap = vertebra_cmap()

    # Focus on the strongest claim: gt_missing_bottom (L5 issue)
    targets = summary[summary["case_class"] == "gt_missing_bottom"].sort_values("mc_dice")
    print(f"verifying {len(targets)} gt_missing_bottom cases (anatomical check)")

    for _, row_sum in targets.iterrows():
        pid = int(row_sum["patient_id"])
        category = row_sum["category"]
        prefix = "S" if category == "Scoliosis" else "N"

        raw_img_path = DATASET / category / f"{prefix}_{pid}.jpg"
        raw_mask_path = DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_{prefix}_{pid}.png"
        raw_img = np.array(Image.open(raw_img_path).convert("L"))
        raw_mask = np.array(Image.open(raw_mask_path))
        H, W = raw_img.shape

        # run prediction
        df_row = clean[(clean["patient_id"] == pid) & (clean["category"] == category)].iloc[0]
        out = predictor.predict_from_row(df_row, tta="hflip")
        pred_512 = out["pred"].cpu().numpy().astype(np.int32)

        # upscale pred to native dimensions for direct anatomical comparison
        pred_native = upscale_pred_to_native(pred_512, H, W)

        # GT centroids at native res
        gt_cents = centroids(raw_mask)
        gt_lowest_y = max(c[1] for c in gt_cents.values())
        gt_lowest_id = max(gt_cents.items(), key=lambda kv: kv[1][1])[0]

        pred_cents = centroids(pred_native)
        pred_lowest_y = max(c[1] for c in pred_cents.values())
        pred_lowest_id = max(pred_cents.items(), key=lambda kv: kv[1][1])[0]

        # crop the bottom 40% of the image (zoom on lumbar/sacrum region)
        crop_top = int(H * 0.55)
        crop = raw_img[crop_top:]
        gt_crop = raw_mask[crop_top:]
        pred_crop = pred_native[crop_top:]

        # crop centroids re-indexed
        gt_cents_crop = {vid: (cx, cy - crop_top) for vid, (cx, cy) in gt_cents.items() if cy >= crop_top}
        pred_cents_crop = {vid: (cx, cy - crop_top) for vid, (cx, cy) in pred_cents.items() if cy >= crop_top}

        fig, axes = plt.subplots(1, 3, figsize=(18, 11))

        # Panel A: raw image alone — for anatomy reading
        axes[0].imshow(crop, cmap="gray", aspect="equal")
        axes[0].axhline(y=gt_lowest_y - crop_top, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
        axes[0].text(5, gt_lowest_y - crop_top - 5, f"GT lowest: {VERT_LABELS[gt_lowest_id-1]}",
                     color="cyan", fontsize=10, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        axes[0].axhline(y=pred_lowest_y - crop_top, color="magenta", linewidth=1, linestyle="--", alpha=0.7)
        axes[0].text(5, pred_lowest_y - crop_top + 18, f"Pred lowest: {VERT_LABELS[pred_lowest_id-1]}",
                     color="magenta", fontsize=10, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        axes[0].set_title(
            f"RAW radiograph (bottom 45%) — {prefix}_{pid}\n"
            f"Cyan line = GT's lowest labeled vertebra centroid\n"
            f"Magenta line = model's lowest predicted vertebra centroid\n"
            f"VERIFY: between cyan & magenta — is there a vertebra shape, or sacrum?",
            fontsize=11,
        )
        axes[0].axis("off")

        # Panel B: GT overlay
        axes[1].imshow(crop, cmap="gray", aspect="equal")
        axes[1].imshow(gt_crop, cmap=cmap, vmin=0, vmax=17, alpha=0.45)
        for vid, (cx, cy) in gt_cents_crop.items():
            axes[1].text(cx, cy, VERT_LABELS[vid - 1],
                         color="yellow", fontsize=11, ha="center", va="center", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, lw=0))
        axes[1].set_title(f"GT overlay (expert-labeled)\nGT stops at {VERT_LABELS[gt_lowest_id-1]}", fontsize=11)
        axes[1].axis("off")

        # Panel C: model prediction overlay (upscaled to native)
        axes[2].imshow(crop, cmap="gray", aspect="equal")
        axes[2].imshow(pred_crop, cmap=cmap, vmin=0, vmax=17, alpha=0.45)
        for vid, (cx, cy) in pred_cents_crop.items():
            color = "red" if vid > gt_lowest_id else "yellow"
            axes[2].text(cx, cy, VERT_LABELS[vid - 1],
                         color=color, fontsize=11, ha="center", va="center", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, lw=0))
        axes[2].set_title(
            f"Model prediction (RED = beyond GT's range)\nModel extends to {VERT_LABELS[pred_lowest_id-1]}",
            fontsize=11,
        )
        axes[2].axis("off")

        cobb = row_sum["cobb_deg"]
        cobb_str = f"Cobb={cobb:.1f}°" if pd.notna(cobb) else "Normal"
        fig.suptitle(
            f"ANATOMY VERIFICATION — {category} {prefix}_{pid}  |  {cobb_str}  |  mc Dice={row_sum['mc_dice']:.3f}\n"
            f"GT has {row_sum['gt_count']} vertebrae, model predicts {row_sum['pred_count']}. "
            f"Question: is the model's red-tagged vertebra sitting on a real vertebral body, or on sacrum/pelvis?",
            fontsize=13, y=0.99,
        )
        fig.tight_layout()
        out_path = OUT_DIR / f"{prefix}_{pid}_anatomy.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {prefix}_{pid}: GT stops at {VERT_LABELS[gt_lowest_id-1]}, "
              f"pred extends to {VERT_LABELS[pred_lowest_id-1]}  ({out_path.name})")

    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
