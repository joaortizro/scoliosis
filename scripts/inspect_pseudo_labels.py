"""Visual + quantitative sanity check on Roboflow pseudo-labels.

Path A debug — Step 1.5 + Step 2 pilot showed pseudo-labels HURT fold-0
Dice by 3.5 percentage points. This script inspects the quality of
accepted pseudo-labels before deciding whether to retry with stricter
thresholds (Step A) or pivot the thesis (Step E).

For each sampled case:
  1. Resize Roboflow image to 512×256 (the inference resolution).
  2. Overlay Roboflow's own human bbox annotations (red boxes).
  3. Overlay the predicted pseudo-mask (colored per vertebra ID).
  4. Compute alignment metrics:
       - distance between Roboflow bbox centroid and pseudo-class centroid
       - per-class IoU between bbox-derived region and pseudo-mask region
  5. Save composite PNG to data/processed/roboflow_pseudo_labels/inspection/

Output: PNGs + a summary CSV (alignment metrics per case).

Usage:
    python scripts/inspect_pseudo_labels.py --n-strict 5 --n-salvage 5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.training.dataset import IMG_H, IMG_W  # noqa: E402  — must come after path insert

REPO_ROOT = Path(__file__).resolve().parents[1]
PSEUDO_ROOT = REPO_ROOT / "data" / "processed" / "roboflow_pseudo_labels"
ROBOFLOW_ROOT = REPO_ROOT / "data" / "raw" / "roboflow_scoliosis_v16"
OUT_DIR = PSEUDO_ROOT / "inspection"

NUM_VERTEBRA_CLASSES = 17


def _vertebra_palette() -> list[tuple[int, int, int]]:
    """17 distinct colors for vertebrae T1..L5, plus black for bg."""
    rng = np.random.default_rng(42)
    base = []
    for i in range(NUM_VERTEBRA_CLASSES + 1):
        if i == 0:
            base.append((0, 0, 0))
        else:
            # Pleasant high-saturation HSV → RGB
            hue = (i * 37) % 360
            s, v = 0.85, 0.95
            c = v * s
            x = c * (1 - abs((hue / 60.0) % 2 - 1))
            m = v - c
            if hue < 60: rgb = (c, x, 0)
            elif hue < 120: rgb = (x, c, 0)
            elif hue < 180: rgb = (0, c, x)
            elif hue < 240: rgb = (0, x, c)
            elif hue < 300: rgb = (x, 0, c)
            else: rgb = (c, 0, x)
            base.append(tuple(int((v + m) * 255) for v in rgb))
    return base


def _load_image_resized(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.array(img)


def _load_roboflow_bboxes(stem: str, split: str) -> list[tuple[float, float, float, float]]:
    label_file = ROBOFLOW_ROOT / "labels" / split / f"{stem}.txt"
    if not label_file.exists():
        return []
    out: list[tuple[float, float, float, float]] = []
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        if cls_id != 0:
            continue
        cx, cy, w, h = (float(p) for p in parts[1:5])
        out.append((cx, cy, w, h))
    return out


def _alignment_metrics(
    pseudo_mask: np.ndarray,
    rf_bboxes: list[tuple[float, float, float, float]],
) -> dict:
    """Match pseudo-mask vertebra centroids to nearest Roboflow bbox; compute mean distance."""
    h, w = pseudo_mask.shape
    pred_centroids: list[tuple[float, float, int]] = []
    for cls_id in range(1, NUM_VERTEBRA_CLASSES + 1):
        ys, xs = np.where(pseudo_mask == cls_id)
        if len(ys) == 0:
            continue
        cy = float(ys.mean()) / h
        cx = float(xs.mean()) / w
        pred_centroids.append((cx, cy, cls_id))

    if not pred_centroids or not rf_bboxes:
        return {
            "n_pred": len(pred_centroids),
            "n_rf": len(rf_bboxes),
            "mean_nearest_dist_norm": float("nan"),
            "median_nearest_dist_norm": float("nan"),
            "max_nearest_dist_norm": float("nan"),
        }

    rf_centroids = np.array([(cx, cy) for (cx, cy, _, _) in rf_bboxes])
    pred_arr = np.array([(cx, cy) for (cx, cy, _) in pred_centroids])
    # Pairwise distances normalized (image diagonal = sqrt(1+1))
    dists = np.linalg.norm(pred_arr[:, None, :] - rf_centroids[None, :, :], axis=-1)
    nearest = dists.min(axis=1)
    return {
        "n_pred": len(pred_centroids),
        "n_rf": len(rf_bboxes),
        "mean_nearest_dist_norm": float(nearest.mean()),
        "median_nearest_dist_norm": float(np.median(nearest)),
        "max_nearest_dist_norm": float(nearest.max()),
    }


def _render_composite(
    image_gray: np.ndarray,
    pseudo_mask: np.ndarray,
    rf_bboxes: list[tuple[float, float, float, float]],
    out_path: Path,
    title: str,
) -> None:
    palette = _vertebra_palette()
    rgb = np.stack([image_gray, image_gray, image_gray], axis=-1).astype(np.uint8)

    # Overlay pseudo-mask (50% blend)
    overlay = np.zeros_like(rgb)
    for cls_id in range(1, NUM_VERTEBRA_CLASSES + 1):
        mask = pseudo_mask == cls_id
        if not mask.any():
            continue
        overlay[mask] = palette[cls_id]
    blended = (0.55 * rgb + 0.45 * overlay).clip(0, 255).astype(np.uint8)
    composite = blended.copy()

    pil = Image.fromarray(composite)
    draw = ImageDraw.Draw(pil)
    h, w = pseudo_mask.shape
    for (cx, cy, bw, bh) in rf_bboxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)

    # Title bar
    bar = Image.new("RGB", (w, 30), (40, 40, 40))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text((6, 8), title, fill=(255, 255, 255))
    final = Image.new("RGB", (w, h + 30))
    final.paste(bar, (0, 0))
    final.paste(pil, (0, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-strict", type=int, default=5)
    ap.add_argument("--n-salvage", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = pd.read_csv(PSEUDO_ROOT / "manifest.csv")
    accepted = manifest[manifest["accepted"] == True].copy()

    strict_only = accepted[
        (accepted["n_vertebrae"] >= 14)
        & (accepted["mean_fg_confidence"] >= 0.70)
    ]
    salvage_only = accepted[~accepted.index.isin(strict_only.index)]

    rng = random.Random(args.seed)
    strict_sample = rng.sample(strict_only.index.tolist(), min(args.n_strict, len(strict_only)))
    salvage_sample = rng.sample(salvage_only.index.tolist(), min(args.n_salvage, len(salvage_only)))

    print(f"Strict-only pool: {len(strict_only)}  Salvage pool: {len(salvage_only)}")
    print(f"Sampling: {len(strict_sample)} strict + {len(salvage_sample)} salvage")
    print()

    rows: list[dict] = []
    for tag, idxs in [("strict", strict_sample), ("salvage", salvage_sample)]:
        for idx in idxs:
            row = manifest.loc[idx]
            stem = row["stem"]
            split = row["split"]
            image_path = ROBOFLOW_ROOT / "images" / split / f"{stem}.jpg"
            mask_path = PSEUDO_ROOT / "masks" / f"{stem}.png"
            if not image_path.exists() or not mask_path.exists():
                print(f"  [SKIP] {stem}: missing files")
                continue
            img = _load_image_resized(image_path)
            mask = np.array(Image.open(mask_path))
            rf_bboxes = _load_roboflow_bboxes(stem, split)
            metrics = _alignment_metrics(mask, rf_bboxes)
            metrics["tag"] = tag
            metrics["stem"] = stem
            metrics["pred_vertebrae"] = int(row["n_vertebrae"])
            metrics["mean_fg_conf"] = float(row["mean_fg_confidence"])
            metrics["roboflow_bbox_count"] = int(row["roboflow_bbox_count"])
            rows.append(metrics)

            title = (
                f"[{tag}] {stem[:30]}  rf_bbox={len(rf_bboxes)} "
                f"pred_vert={metrics['n_pred']}  "
                f"conf={metrics['mean_fg_conf']:.2f}  "
                f"mean_dist={metrics['mean_nearest_dist_norm']:.3f}"
            )
            out_path = OUT_DIR / f"{tag}_{stem}.png"
            _render_composite(img, mask, rf_bboxes, out_path, title)
            print(f"  [{tag}] {stem}: n_pred={metrics['n_pred']} "
                  f"rf={metrics['n_rf']} mean_dist={metrics['mean_nearest_dist_norm']:.3f} "
                  f"median_dist={metrics['median_nearest_dist_norm']:.3f} "
                  f"max_dist={metrics['max_nearest_dist_norm']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "alignment_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print()
    print("=== summary by tag ===")
    print(df.groupby("tag")[["mean_nearest_dist_norm", "median_nearest_dist_norm",
                              "max_nearest_dist_norm", "n_pred", "n_rf"]].agg(["mean", "std"]))
    print(f"\nPNGs in: {OUT_DIR}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
