"""Single-PNG visibility sweep for the L5-bottom and T1-top cases.

For each case where model predicts a vertebra outside GT's range, render
a side-by-side view of:
  - left: raw radiograph cropped to the contested region (anatomy reference)
  - right: same crop with the model's "extra" prediction overlaid in red

Goal: human scans in 2-3 minutes and tags each case as either
  - "GT incomplete, vertebra visible" (hand-annotation needed)
  - "Model hallucination, no vertebra there" (training fix needed)

Output: one PNG per group (L5 sweep, T1 sweep) + a stub CSV the user
can fill in the verdict column.
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
OUT_DIR = REPO / "notebooks/sandbox/viz_2026-05-11/visibility_sweep"

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


def upscale_pred(pred_512x256: np.ndarray, native_h: int, native_w: int) -> np.ndarray:
    img = Image.fromarray(pred_512x256.astype(np.uint8))
    return np.array(img.resize((native_w, native_h), Image.NEAREST)).astype(np.int32)


def build_sweep(
    cases: list[tuple[str, int, str]],   # (category, pid, kind)
    side: str,                           # "bottom" or "top"
    title: str,
    out_path: Path,
    predictor: Predictor,
    clean: pd.DataFrame,
    cmap: ListedColormap,
) -> list[dict]:
    """Build a multi-row sweep PNG for a list of cases."""
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    rows = []
    for i, (category, pid, kind) in enumerate(cases):
        prefix = "S" if category == "Scoliosis" else "N"
        raw_img = np.array(Image.open(DATASET / category / f"{prefix}_{pid}.jpg").convert("L"))
        raw_mask = np.array(Image.open(DATASET / "LabelMultiClass_ID_PNG" / f"LabelMulti_{prefix}_{pid}.png"))
        H, W = raw_img.shape

        df_row = clean[(clean["patient_id"] == pid) & (clean["category"] == category)].iloc[0]
        out = predictor.predict_from_row(df_row, tta="hflip")
        pred_native = upscale_pred(out["pred"].cpu().numpy().astype(np.int32), H, W)

        gt_cents = centroids(raw_mask)
        pred_cents = centroids(pred_native)
        gt_ids = sorted(gt_cents.keys())
        pred_ids = sorted(pred_cents.keys())
        extra_ids = sorted(set(pred_ids) - set(gt_ids))

        # Crop region for inspection
        if side == "bottom":
            # Anchor at the last GT vertebra; show 25% above + 30% of remaining image below
            anchor_y = gt_cents[max(gt_ids)][1]
            crop_top = max(0, int(anchor_y) - int(H * 0.10))
            crop_bot = min(H, int(anchor_y) + int(H * 0.25))
        else:  # top
            anchor_y = gt_cents[min(gt_ids)][1]
            crop_top = max(0, int(anchor_y) - int(H * 0.25))
            crop_bot = min(H, int(anchor_y) + int(H * 0.10))

        raw_crop = raw_img[crop_top:crop_bot]
        pred_crop = pred_native[crop_top:crop_bot]

        # Build a mask that ONLY shows the model's extra IDs (so user sees just the contested predictions)
        extras_only = np.where(np.isin(pred_crop, extra_ids), pred_crop, 0)

        gt_anchor_y_local = gt_cents[max(gt_ids) if side == "bottom" else min(gt_ids)][1] - crop_top
        gt_anchor_label = VERT_LABELS[(max(gt_ids) if side == "bottom" else min(gt_ids)) - 1]

        # Left column: raw only — anatomy reference
        axes[i, 0].imshow(raw_crop, cmap="gray", aspect="equal")
        axes[i, 0].axhline(y=gt_anchor_y_local, color="cyan", linewidth=1.5, linestyle="--", alpha=0.85)
        axes[i, 0].text(5, gt_anchor_y_local - 8, f"GT's {gt_anchor_label}",
                        color="cyan", fontsize=11, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))
        cobb = df_row.get("cobb_angle_deg")
        cobb_str = f"Cobb={cobb:.1f}°" if pd.notna(cobb) else "Normal"
        axes[i, 0].set_title(
            f"{category} {prefix}_{pid}  |  {cobb_str}\n"
            f"GT range: {VERT_LABELS[min(gt_ids)-1]}..{VERT_LABELS[max(gt_ids)-1]}  ({len(gt_ids)} vertebrae)",
            fontsize=11,
        )
        axes[i, 0].axis("off")

        # Right column: same crop with the model's extra IDs overlaid
        axes[i, 1].imshow(raw_crop, cmap="gray", aspect="equal")
        if extras_only.any():
            axes[i, 1].imshow(extras_only, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
            for vid in extra_ids:
                if vid in pred_cents:
                    cx, cy = pred_cents[vid]
                    cy_local = cy - crop_top
                    if 0 <= cy_local < (crop_bot - crop_top):
                        axes[i, 1].text(cx, cy_local, VERT_LABELS[vid - 1],
                                        color="red", fontsize=14, weight="bold",
                                        ha="center", va="center",
                                        bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                                                  alpha=0.75, lw=0))
        axes[i, 1].axhline(y=gt_anchor_y_local, color="cyan", linewidth=1.5, linestyle="--", alpha=0.85)
        axes[i, 1].set_title(
            f"Model extras (RED) = {[VERT_LABELS[v-1] for v in extra_ids]}\n"
            f"VERIFY: vertebra visible at red position? → 'GT incomplete'  |  bone shadow only? → 'hallucination'",
            fontsize=11,
        )
        axes[i, 1].axis("off")

        rows.append({
            "patient_id": pid,
            "category": category,
            "kind": kind,
            "cobb_deg": cobb,
            "gt_range": f"{VERT_LABELS[min(gt_ids)-1]}..{VERT_LABELS[max(gt_ids)-1]}",
            "extras": ",".join(VERT_LABELS[v-1] for v in extra_ids),
            "verdict": "",   # user fills this in
        })

    fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads(SENTINEL.read_text())
    predictor = Predictor(REPO / sentinel["folds"][4]["run_dir"], device=torch.device("cpu"))
    clean = pd.read_csv(CLEAN_INDEX)
    summary = pd.read_csv(SUMMARY)
    cmap = vertebra_cmap()

    # L5 bottom sweep — 9 cases
    bottom_cases = [
        (r["category"], int(r["patient_id"]), "bottom")
        for _, r in summary[summary["case_class"] == "gt_missing_bottom"].sort_values("mc_dice").iterrows()
    ]
    bottom_rows = build_sweep(
        bottom_cases, "bottom",
        title="L5-VISIBILITY SWEEP — 9 cases where model predicts L5 but GT stops at L4\n"
              "For each row: scan the right panel. Is there a vertebra-shaped bone where the red 'L5' label sits?\n"
              "  YES → 'incomplete' (hand-annotate)  |  NO (sacrum/pelvis only) → 'hallucination' (training fix)",
        out_path=OUT_DIR / "L5_visibility_sweep.png",
        predictor=predictor, clean=clean, cmap=cmap,
    )

    # T1 top sweep — 2 cases
    top_cases = [
        (r["category"], int(r["patient_id"]), "top")
        for _, r in summary[summary["case_class"] == "gt_missing_top"].sort_values("mc_dice").iterrows()
    ]
    if top_cases:
        top_rows = build_sweep(
            top_cases, "top",
            title="T1-TOP SWEEP — 2 cases where model predicts vertebrae above GT's topmost label\n"
                  "Same question: real vertebra above the cyan line, or model bleeding into clavicle/cervical?",
            out_path=OUT_DIR / "T1_visibility_sweep.png",
            predictor=predictor, clean=clean, cmap=cmap,
        )
    else:
        top_rows = []

    # Combined CSV stub for user verdicts
    df = pd.DataFrame(bottom_rows + top_rows)
    csv_path = OUT_DIR / "verdicts_TO_FILL.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV stub for verdicts: {csv_path}")
    print("After scanning the PNGs, fill the 'verdict' column with: incomplete | hallucination | ambiguous")


if __name__ == "__main__":
    main()
