"""Anatomy verification for gt_missing_middle cases.

Shows the FULL spine at native resolution with:
  - GT: explicit labels at each centroid; missing vertebra ID highlighted by absence
  - Pred: model's labels; the "extra" mid-spine ID shown in red
  - The expected y-position of the missing vertebra interpolated from neighbors

Strong evidence for v2 GT having mid-spine annotation gaps that
cannot be explained by FOV cropping (since vertebrae above AND
below are labeled).
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
OUT_DIR = REPO / "notebooks/sandbox/viz_2026-05-11/anatomy_verify_middle"
VERT_LABELS = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]

CASES = [
    ("Scoliosis", 150, 6),   # T6 missing
    ("Scoliosis", 41, 14),   # L2 missing (id=14)
    ("Normal", 26, 5),       # T5 missing
    ("Normal", 36, 10),      # T10 missing
    ("Normal", 52, 10),      # T10 missing
]


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


def upscale_pred(pred_512x256: np.ndarray, native_h: int, native_w: int) -> np.ndarray:
    img = Image.fromarray(pred_512x256.astype(np.uint8))
    return np.array(img.resize((native_w, native_h), Image.NEAREST)).astype(np.int32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads(SENTINEL.read_text())
    predictor = Predictor(REPO / sentinel["folds"][4]["run_dir"], device=torch.device("cpu"))
    clean = pd.read_csv(CLEAN_INDEX)
    cmap = vertebra_cmap()

    for category, pid, missing_id in CASES:
        prefix = "S" if category == "Scoliosis" else "N"
        raw_img = np.array(Image.open(DATASET / category / f"{prefix}_{pid}.jpg").convert("L"))
        raw_mask = np.array(Image.open(DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_{prefix}_{pid}.png"))
        H, W = raw_img.shape

        df_row = clean[(clean["patient_id"] == pid) & (clean["category"] == category)].iloc[0]
        out = predictor.predict_from_row(df_row, tta="hflip")
        pred_native = upscale_pred(out["pred"].cpu().numpy().astype(np.int32), H, W)

        gt_cents = centroids(raw_mask)
        pred_cents = centroids(pred_native)

        # Predict expected y-position of missing vertebra from neighbors in GT
        prev_id, next_id = missing_id - 1, missing_id + 1
        if prev_id in gt_cents and next_id in gt_cents:
            expected_y = (gt_cents[prev_id][1] + gt_cents[next_id][1]) / 2
            expected_x = (gt_cents[prev_id][0] + gt_cents[next_id][0]) / 2
        else:
            expected_y = expected_x = None

        # Tight crop around the missing region: ±200 px on the y-axis
        if expected_y:
            crop_top = max(0, int(expected_y) - 220)
            crop_bot = min(H, int(expected_y) + 220)
            crop = raw_img[crop_top:crop_bot]
            gt_crop = raw_mask[crop_top:crop_bot]
            pred_crop = pred_native[crop_top:crop_bot]
            cents_crop = lambda d: {v: (cx, cy - crop_top) for v, (cx, cy) in d.items() if crop_top <= cy <= crop_bot}
        else:
            crop, gt_crop, pred_crop = raw_img, raw_mask, pred_native
            cents_crop = lambda d: d

        gt_cents_c = cents_crop(gt_cents)
        pred_cents_c = cents_crop(pred_cents)
        expected_y_c = expected_y - crop_top if expected_y else None

        fig, axes = plt.subplots(1, 3, figsize=(18, 11))

        # Panel A: raw with predicted gap marker
        axes[0].imshow(crop, cmap="gray", aspect="equal")
        if expected_y_c is not None:
            axes[0].axhline(y=expected_y_c, color="lime", linewidth=2, linestyle="--", alpha=0.8)
            axes[0].text(5, expected_y_c - 8, f"expected position of missing {VERT_LABELS[missing_id-1]}",
                         color="lime", fontsize=12, weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75))
        axes[0].set_title(
            f"RAW (zoomed around missing region, ±220 px)\n"
            f"GREEN line = expected position of {VERT_LABELS[missing_id-1]} (interpolated from GT neighbors)\n"
            f"VERIFY: is there a vertebra-shaped bone at the green line?",
            fontsize=11,
        )
        axes[0].axis("off")

        # Panel B: GT overlay with neighbor labels
        axes[1].imshow(crop, cmap="gray", aspect="equal")
        axes[1].imshow(gt_crop, cmap=cmap, vmin=0, vmax=17, alpha=0.45)
        for vid, (cx, cy) in gt_cents_c.items():
            color = "lime" if vid in (missing_id - 1, missing_id + 1) else "yellow"
            axes[1].text(cx, cy, VERT_LABELS[vid - 1],
                         color=color, fontsize=12, ha="center", va="center", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, lw=0))
        if expected_y_c is not None:
            axes[1].axhline(y=expected_y_c, color="lime", linewidth=1, linestyle="--", alpha=0.6)
        axes[1].set_title(
            f"GT (expert-labeled) — {VERT_LABELS[missing_id-1]} ABSENT\n"
            f"green labels = neighbors of the missing ID  |  green line = where {VERT_LABELS[missing_id-1]} should be",
            fontsize=11,
        )
        axes[1].axis("off")

        # Panel C: model prediction
        axes[2].imshow(crop, cmap="gray", aspect="equal")
        axes[2].imshow(pred_crop, cmap=cmap, vmin=0, vmax=17, alpha=0.45)
        for vid, (cx, cy) in pred_cents_c.items():
            color = "red" if vid == missing_id else "yellow"
            axes[2].text(cx, cy, VERT_LABELS[vid - 1],
                         color=color, fontsize=12, ha="center", va="center", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, lw=0))
        axes[2].set_title(
            f"Model prediction — RED label = the {VERT_LABELS[missing_id-1]} that GT lacks",
            fontsize=11,
        )
        axes[2].axis("off")

        cobb = df_row.get("cobb_angle_deg")
        cobb_str = f"Cobb={cobb:.1f}°" if pd.notna(cobb) else "Normal"
        fig.suptitle(
            f"MID-SPINE GAP — {category} {prefix}_{pid}  |  {cobb_str}\n"
            f"GT has T1..L5 BUT skips {VERT_LABELS[missing_id-1]} (between {VERT_LABELS[missing_id-2]} and {VERT_LABELS[missing_id]}).\n"
            f"Both neighbors are clearly labeled in GT → cannot be FOV-cropping → annotation gap.",
            fontsize=13, y=0.99,
        )
        fig.tight_layout()
        out_path = OUT_DIR / f"{prefix}_{pid}_missing_{VERT_LABELS[missing_id-1]}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {prefix}_{pid}: missing {VERT_LABELS[missing_id-1]} → {out_path.name}")


if __name__ == "__main__":
    main()
