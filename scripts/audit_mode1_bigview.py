"""Big per-case view for Mode 1 (count-mismatch) cases.

For each case, render a wide 3-panel figure:
  Left  : raw radiograph at native resolution (count vertebrae by eye)
  Middle: GT mask overlay with vertebra IDs labeled at each centroid
  Right : model prediction overlay with vertebra IDs labeled

Goal: human counts visible vertebrae in the image. If the model
predicted count matches what's visible, GT is incomplete (hand-annotate).
If the GT count matches what's visible, model is hallucinating
(train-time fix).
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
EVAL_CSV = REPO / "notebooks/sandbox/viz_2026-05-11/fold4_partial_coverage_eval.csv"
OUT_DIR = REPO / "notebooks/sandbox/viz_2026-05-11/mode1_bigview"

# Five Mode-1 / count-mismatch cases from fold 4 worst-9
CASES = [
    ("Scoliosis", 200, "16 GT vs 17 pred — mc=0.001"),
    ("Scoliosis", 105, "16 GT vs 17 pred — mc=0.022"),
    ("Scoliosis", 38, "14 GT vs 15 pred — mc=0.399"),
    ("Scoliosis", 150, "16 GT vs 17 pred — mc=0.445"),
    ("Scoliosis", 136, "16 GT vs 17 pred — mc=0.463"),
]

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


def label_vertebrae(ax, cents, fontsize=11):
    for vid, (cx, cy) in cents.items():
        ax.text(cx, cy, VERT_LABELS[vid - 1],
                color="yellow", fontsize=fontsize, ha="center", va="center",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.65, lw=0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads(SENTINEL.read_text())
    predictor = Predictor(REPO / sentinel["folds"][4]["run_dir"], device=torch.device("cpu"))
    clean = pd.read_csv(CLEAN_INDEX)
    cmap = vertebra_cmap()

    for category, pid, blurb in CASES:
        prefix = "S" if category == "Scoliosis" else "N"
        raw_img_path = DATASET / category / f"{prefix}_{pid}.jpg"
        raw_mask_path = DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_{prefix}_{pid}.png"
        if not raw_img_path.exists():
            print(f"missing {raw_img_path}")
            continue

        raw_img = np.array(Image.open(raw_img_path).convert("L"))
        raw_mask = np.array(Image.open(raw_mask_path))
        raw_cents = centroids(raw_mask)

        # model prediction at trainer resolution (512x256), then upscale display only
        row = clean[(clean["patient_id"] == pid) & (clean["category"] == category)].iloc[0]
        out = predictor.predict_from_row(row, tta="hflip")
        pred_img = out["image"].squeeze().cpu().numpy()  # 512x256
        pred = out["pred"].cpu().numpy().astype(np.int32)
        pred_cents = centroids(pred)
        gt_512 = out["seg"].cpu().numpy().astype(np.int32)

        gt_ids_in_raw = sorted(int(u) for u in np.unique(raw_mask) if u > 0)
        gt_ids_512 = sorted(int(u) for u in np.unique(gt_512) if u > 0)
        pred_ids = sorted(pred_cents.keys())

        # Figure: 1 row x 3 cols (raw native | GT-at-trainer-res | pred-at-trainer-res)
        # All three panels at trainer res for fair visual comparison.
        fig, axes = plt.subplots(1, 3, figsize=(15, 12))

        # Col 0: raw radiograph (native resolution) with GT label IDs
        axes[0].imshow(raw_img, cmap="gray", aspect="equal")
        axes[0].imshow(raw_mask, cmap=cmap, vmin=0, vmax=17, alpha=0.30)
        label_vertebrae(axes[0], raw_cents, fontsize=12)
        axes[0].set_title(
            f"RAW radiograph + GT labels (native res {raw_img.shape[1]}×{raw_img.shape[0]} px)\n"
            f"GT covers IDs {gt_ids_in_raw} → top:{VERT_LABELS[gt_ids_in_raw[0]-1]} "
            f"bot:{VERT_LABELS[gt_ids_in_raw[-1]-1]} ({len(gt_ids_in_raw)} vertebrae)",
            fontsize=10,
        )
        axes[0].axis("off")

        # Col 1: GT at trainer resolution
        axes[1].imshow(pred_img, cmap="gray", aspect="equal")
        axes[1].imshow(gt_512, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        label_vertebrae(axes[1], centroids(gt_512), fontsize=11)
        axes[1].set_title(
            f"GT at trainer res (512×256)\n{len(gt_ids_512)} vertebrae",
            fontsize=10,
        )
        axes[1].axis("off")

        # Col 2: model prediction at trainer resolution
        axes[2].imshow(pred_img, cmap="gray", aspect="equal")
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        label_vertebrae(axes[2], pred_cents, fontsize=11)
        axes[2].set_title(
            f"Model prediction (Phase 1.2 fold 4)\n{len(pred_ids)} vertebrae predicted",
            fontsize=10,
        )
        axes[2].axis("off")

        fig.suptitle(
            f"{category} {prefix}_{pid}  —  {blurb}\n"
            f"Question: count visible vertebrae in left panel. "
            f"Match GT ({len(gt_ids_in_raw)}) or model ({len(pred_ids)})?",
            fontsize=13, y=0.995,
        )
        fig.tight_layout()
        out_path = OUT_DIR / f"{prefix}_{pid}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)  "
              f"GT={len(gt_ids_in_raw)} pred={len(pred_ids)}")

    print(f"\noutput dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
