"""Spatial-constraint refinement of Roboflow pseudo-labels.

Uses Roboflow's own human bbox annotations as a spatial mask: zero out
any predicted vertebra-class pixels OUTSIDE the union of Roboflow
bboxes (with 10% padding). Eliminates the failure modes we saw in
visual inspection:
  - DICOM viewer UI labeled as vertebrae
  - Skull / upper-chest regions labeled
  - Sacrum (S1) bleed below L5
  - Stray fragments in lung area

The pseudo-mask CLASSES are kept the same INSIDE the spine zone — we
trust the model's predicted vertebra-ID assignment, just constrain it
spatially.

After spatial masking, re-apply the original quality filter (≥14 distinct
vertebrae, mean fg conf ≥0.70, fg_frac ∈ [0.005, 0.40]). Some labels
that were borderline may now fall below the threshold and get rejected.

Output: a NEW pseudo-label dir + manifest.

Usage:
    python scripts/spatial_constraint_filter.py --pad-frac 0.10 --strict-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.detection.roboflow_filter import count_vertebrae_in_label_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "data" / "processed" / "roboflow_pseudo_labels"
ROBOFLOW_ROOT = REPO_ROOT / "data" / "raw" / "roboflow_scoliosis_v16"

MIN_VERTEBRAE_FOR_PSEUDO_LABEL = 14
MIN_MEAN_CONFIDENCE = 0.70
MIN_FG_FRAC = 0.005
MAX_FG_FRAC = 0.40


def _load_roboflow_bboxes(stem: str, split: str) -> list[tuple[float, float, float, float]]:
    label_file = ROBOFLOW_ROOT / "labels" / split / f"{stem}.txt"
    if not label_file.exists():
        return []
    out: list[tuple[float, float, float, float]] = []
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        if int(parts[0]) != 0:
            continue
        cx, cy, w, h = (float(p) for p in parts[1:5])
        out.append((cx, cy, w, h))
    return out


def _spine_zone_mask(
    bboxes: list[tuple[float, float, float, float]],
    h: int,
    w: int,
    pad_frac: float = 0.10,
) -> np.ndarray:
    """Build a binary (H, W) mask covering the union of Roboflow bboxes + padding.

    pad_frac is applied as a fraction of the bbox spine-zone's extent (not
    per-bbox), so the padded zone hugs the full spine column.
    """
    if not bboxes:
        return np.zeros((h, w), dtype=bool)
    # Compute the bounding hull of all bboxes (normalized coords)
    xs_min = min(cx - bw / 2 for (cx, _, bw, _) in bboxes)
    xs_max = max(cx + bw / 2 for (cx, _, bw, _) in bboxes)
    ys_min = min(cy - bh / 2 for (_, cy, _, bh) in bboxes)
    ys_max = max(cy + bh / 2 for (_, cy, _, bh) in bboxes)
    # Pad
    pad_x = pad_frac * (xs_max - xs_min)
    pad_y = pad_frac * (ys_max - ys_min)
    x0 = max(0.0, xs_min - pad_x)
    x1 = min(1.0, xs_max + pad_x)
    y0 = max(0.0, ys_min - pad_y)
    y1 = min(1.0, ys_max + pad_y)
    # Convert to pixel coords
    px0, px1 = int(x0 * w), int(x1 * w)
    py0, py1 = int(y0 * h), int(y1 * h)
    mask = np.zeros((h, w), dtype=bool)
    mask[py0:py1, px0:px1] = True
    return mask


def _quality_filter(mask: np.ndarray, conf_map: np.ndarray) -> tuple[bool, str]:
    nonzero_classes = set(np.unique(mask).tolist()) - {0}
    n_vert = len(nonzero_classes)
    if n_vert < MIN_VERTEBRAE_FOR_PSEUDO_LABEL:
        return False, f"only {n_vert} vertebrae after spatial mask (need >= {MIN_VERTEBRAE_FOR_PSEUDO_LABEL})"
    fg_mask = mask > 0
    fg_count = int(fg_mask.sum())
    fg_frac = fg_count / mask.size
    if fg_frac < MIN_FG_FRAC:
        return False, f"fg fraction {fg_frac:.4f} below {MIN_FG_FRAC}"
    if fg_frac > MAX_FG_FRAC:
        return False, f"fg fraction {fg_frac:.4f} above {MAX_FG_FRAC}"
    if fg_count == 0:
        return False, "no foreground pixels"
    mean_conf = float(conf_map[fg_mask].mean())
    if mean_conf < MIN_MEAN_CONFIDENCE:
        return False, f"mean fg confidence {mean_conf:.3f} below {MIN_MEAN_CONFIDENCE}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=SRC_ROOT)
    ap.add_argument("--out-suffix", default="_spatial",
                    help="appended to roboflow_pseudo_labels dir name")
    ap.add_argument("--pad-frac", type=float, default=0.10,
                    help="padding around the union of Roboflow bboxes")
    ap.add_argument("--strict-only", action="store_true",
                    help="process only previously-strict-accepted labels (no bbox-salvage)")
    args = ap.parse_args()

    out_root = REPO_ROOT / "data" / "processed" / f"roboflow_pseudo_labels{args.out_suffix}"
    masks_out = out_root / "masks"
    conf_out = out_root / "confidence"
    images_out = out_root / "images"
    for d in (masks_out, conf_out, images_out):
        d.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.src_dir / "manifest.csv")
    accepted_src = manifest[manifest["accepted"] == True].copy()
    if args.strict_only:
        accepted_src = accepted_src[
            (accepted_src["n_vertebrae"] >= 14)
            & (accepted_src["mean_fg_confidence"] >= 0.70)
        ].copy()
    print(f"Source accepted labels: {len(accepted_src)} (strict_only={args.strict_only})")

    rows: list[dict] = []
    reject_counter: Counter[str] = Counter()
    n_kept = 0
    for _, r in accepted_src.iterrows():
        stem = r["stem"]
        split = r["split"]
        mask_path = args.src_dir / "masks" / f"{stem}.png"
        conf_path = args.src_dir / "confidence" / f"{stem}.npy"
        if not mask_path.exists() or not conf_path.exists():
            reject_counter["missing_artifact"] += 1
            continue
        mask = np.array(Image.open(mask_path))
        conf = np.load(conf_path)
        h, w = mask.shape

        rf_bboxes = _load_roboflow_bboxes(stem, split)
        if not rf_bboxes:
            reject_counter["no_roboflow_bboxes"] += 1
            continue
        zone = _spine_zone_mask(rf_bboxes, h, w, pad_frac=args.pad_frac)
        # Zero out predictions outside the spine zone
        mask_constrained = np.where(zone, mask, 0).astype(np.uint8)
        conf_constrained = np.where(zone, conf, 0.0).astype(np.float32)

        # Pre-constrain stats
        nonzero_pre = set(np.unique(mask).tolist()) - {0}
        nonzero_post = set(np.unique(mask_constrained).tolist()) - {0}
        fg_pre = int((mask > 0).sum())
        fg_post = int((mask_constrained > 0).sum())
        # Fraction of foreground pixels that were OUTSIDE the spine zone — these are
        # the "skull / UI / sacrum bleed" false positives that this filter removes.
        outside_fg_frac = (fg_pre - fg_post) / max(1, fg_pre)

        accepted, reason = _quality_filter(mask_constrained, conf_constrained)
        rows.append({
            "stem": stem,
            "split": split,
            "accepted": accepted,
            "reject_reason": reason,
            "n_vertebrae_pre": len(nonzero_pre),
            "n_vertebrae_post": len(nonzero_post),
            "fg_pixels_pre": fg_pre,
            "fg_pixels_post": fg_post,
            "outside_zone_fg_frac": outside_fg_frac,
            "roboflow_bbox_count": len(rf_bboxes),
        })

        if accepted:
            n_kept += 1
            Image.fromarray(mask_constrained).save(masks_out / f"{stem}.png")
            np.save(conf_out / f"{stem}.npy", conf_constrained)
            src_img = ROBOFLOW_ROOT / "images" / split / f"{stem}.jpg"
            dst_img = images_out / f"{stem}.jpg"
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            if src_img.exists():
                dst_img.symlink_to(src_img.resolve())
        else:
            reject_counter[reason.split(" ")[0]] += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifest.csv", index=False)

    stats = {
        "src_dir": str(args.src_dir),
        "out_dir": str(out_root),
        "n_input": len(accepted_src),
        "n_kept": n_kept,
        "n_rejected_after_spatial": len(accepted_src) - n_kept,
        "rejection_breakdown": dict(reject_counter),
        "pad_frac": args.pad_frac,
        "strict_only": args.strict_only,
        "mean_outside_zone_fg_frac": float(df["outside_zone_fg_frac"].mean()),
        "median_outside_zone_fg_frac": float(df["outside_zone_fg_frac"].median()),
        "max_outside_zone_fg_frac": float(df["outside_zone_fg_frac"].max()),
        "vertebra_lost_per_label_mean": float((df["n_vertebrae_pre"] - df["n_vertebrae_post"]).mean()),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n=== spatial-constraint filter complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
