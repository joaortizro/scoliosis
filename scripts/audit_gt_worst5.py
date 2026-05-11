"""Audit the 5 worst-scoring val cases for GT label correctness.

For each case render: raw radiograph | GT multi-class mask (with vertebra
IDs labeled at centroids) | Cobb overlay (if available) | model prediction
(from inference cache). Lets a human verify whether GT itself is upside-down
or misordered — which would explain why fold 4's mc Dice collapses on these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.inference.predictor import Predictor

DATASET = REPO / "data/raw/Scoliosis_Dataset_v2_corrected"
SENTINEL = REPO / "experiments/results/phase1_2_5fold.json"
CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"

WORST_IDS = [200, 105, 21, 54, 24, 47, 38, 150, 136]
VERT_LABELS = (
    [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
)  # IDs 1..17 → T1..L5


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


def main() -> None:
    import json
    sentinel = json.loads(SENTINEL.read_text())
    run_dir = REPO / sentinel["folds"][4]["run_dir"]
    print(f"loading fold 4 from {run_dir.name}")
    import torch
    predictor = Predictor(run_dir, device=torch.device("cpu"))

    clean = pd.read_csv(CLEAN_INDEX)
    clean["patient_id"] = clean["patient_id"].astype(int)

    cmap = vertebra_cmap()
    n = len(WORST_IDS)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n))

    leaderboard = pd.read_csv(REPO / "notebooks/sandbox/viz_2026-05-11/fold4_all_leaderboard.csv")

    for i, pid in enumerate(WORST_IDS):
        img_path = DATASET / "Scoliosis" / f"S_{pid}.jpg"
        mask_path = DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_S_{pid}.png"
        cobb_overlay_path = DATASET / "RadiographMetrics/overlays" / f"overlay_cobb_recalc_{pid}.png"

        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path))
        unique_ids = sorted(int(u) for u in np.unique(mask) if u > 0)
        cents = centroids(mask)
        # what ID is at the TOP of the image vs BOTTOM
        ids_by_y = sorted(cents.items(), key=lambda kv: kv[1][1])
        topmost_id = ids_by_y[0][0] if ids_by_y else None
        bottommost_id = ids_by_y[-1][0] if ids_by_y else None

        row = leaderboard[leaderboard["patient_id"] == pid].iloc[0]
        cobb = row["cobb_deg"]
        d_bin = row["binary_dice"]
        d_mc = row["mc_dice"]

        # diagnostic: is GT correctly oriented? Top should be T1 (id=1), bottom should be L5 (id=17)
        gt_orientation_ok = (topmost_id == min(unique_ids)) and (bottommost_id == max(unique_ids))
        flip_marker = "✓ T1→L5 top-to-bottom" if gt_orientation_ok else "✗ ORIENTATION SUSPECT"

        # Col 0: raw radiograph
        axes[i, 0].imshow(img, cmap="gray", aspect="equal")
        axes[i, 0].set_title(
            f"S_{pid}  Cobb={cobb:.1f}°\n"
            f"binary={d_bin:.3f}  mc={d_mc:.3f}\n"
            f"GT IDs present: {unique_ids}",
            fontsize=10,
        )
        axes[i, 0].axis("off")

        # Col 1: GT mask with vertebra ID labels at centroids
        axes[i, 1].imshow(img, cmap="gray", aspect="equal")
        axes[i, 1].imshow(mask, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        for vid, (cx, cy) in cents.items():
            label = VERT_LABELS[vid - 1]
            axes[i, 1].text(
                cx, cy, label,
                color="white", fontsize=9, ha="center", va="center",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0),
            )
        topmost_label = VERT_LABELS[topmost_id - 1] if topmost_id else "?"
        bottommost_label = VERT_LABELS[bottommost_id - 1] if bottommost_id else "?"
        axes[i, 1].set_title(
            f"GT labels (top→bot in image: {topmost_label}→{bottommost_label})\n{flip_marker}",
            fontsize=10,
            color="green" if gt_orientation_ok else "red",
        )
        axes[i, 1].axis("off")

        # Col 2: Cobb overlay (the dataset's own diagnostic image)
        if cobb_overlay_path.exists():
            cobb_img = np.array(Image.open(cobb_overlay_path).convert("RGB"))
            axes[i, 2].imshow(cobb_img, aspect="equal")
            axes[i, 2].set_title(f"Cobb overlay (curve fit)\nGT angle: {cobb:.1f}°", fontsize=10)
        else:
            axes[i, 2].text(0.5, 0.5, "no Cobb overlay\n(normal case?)", ha="center", va="center")
            axes[i, 2].set_title("Cobb overlay")
        axes[i, 2].axis("off")

        # Col 3: Model prediction at TRAINER resolution (so we see what the model 'saw')
        idx_row = clean[clean["patient_id"] == pid].iloc[0]
        out = predictor.predict_from_row(idx_row, tta="hflip")
        pred_img = out["image"].squeeze().cpu().numpy()
        pred = out["pred"].cpu().numpy().astype(np.int32)
        pred_cents = centroids(pred)
        axes[i, 3].imshow(pred_img, cmap="gray", aspect="equal")
        axes[i, 3].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        for vid, (cx, cy) in pred_cents.items():
            axes[i, 3].text(
                cx, cy, VERT_LABELS[vid - 1],
                color="white", fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, lw=0),
            )
        pred_ids = sorted(pred_cents.keys())
        pred_top = VERT_LABELS[pred_ids[0] - 1] if pred_ids else "?"
        pred_bot = VERT_LABELS[pred_ids[-1] - 1] if pred_ids else "?"
        # do GT and pred agree on orientation?
        same_orient = (
            (topmost_id is not None) and (pred_ids)
            and ((topmost_id <= 9) == (pred_ids[0] <= 9))  # both small or both big at top
        )
        agree_marker = "agrees w/ GT" if same_orient else "DISAGREES w/ GT"
        axes[i, 3].set_title(
            f"Model pred (top→bot: {pred_top}→{pred_bot}) — {agree_marker}",
            fontsize=10,
            color="green" if same_orient else "orange",
        )
        axes[i, 3].axis("off")

        print(f"S_{pid}: GT top={topmost_label} bot={bottommost_label}  "
              f"pred top={pred_top} bot={pred_bot}  cobb={cobb:.1f}°  "
              f"mc={d_mc:.3f}  GT_orient={'OK' if gt_orientation_ok else 'SUSPECT'}")

    fig.suptitle(
        "GT label audit — 5 worst-scoring fold-4 val cases\n"
        "If 'GT IDs top→bot' shows L*→T* instead of T*→L*, the GT itself is upside-down.",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    out_path = REPO / "notebooks/sandbox/viz_2026-05-11/audit_gt_worst9.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
