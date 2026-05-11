"""Visual inspection of Phase 1.2 best model on held-out val cases.

Picks fold 4 (best single fold, val Dice 0.7189) and samples N cases
from its val split. Saves a composite PNG: image | GT multi-class mask
| predicted mask | overlay-on-image, per case, with per-case Dice.

Usage:
    python scripts/inspect_predictions.py --n 6 --fold 4 --out /tmp/inspect.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.inference.predictor import Predictor
from ai.training.splits import make_cv_folds, trainable_rows

CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"
TEST_HOLDOUT = REPO / "data/processed/audit_v2_corrected/test_holdout.csv"
SENTINEL = REPO / "experiments/results/phase1_2_5fold.json"


def vertebra_cmap() -> ListedColormap:
    base = plt.colormaps["tab20"].colors + plt.colormaps["tab20b"].colors
    colors = [(0, 0, 0, 0)] + [base[i % len(base)] for i in range(17)]
    return ListedColormap(colors, name="vert18")


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_fg = pred > 0
    gt_fg = gt > 0
    inter = (pred_fg & gt_fg).sum()
    denom = pred_fg.sum() + gt_fg.sum()
    return float(2 * inter / denom) if denom else 1.0


def per_class_dice(pred: np.ndarray, gt: np.ndarray, n_classes: int = 18) -> list[float]:
    out = []
    for c in range(1, n_classes):
        p = pred == c
        g = gt == c
        denom = p.sum() + g.sum()
        out.append(float(2 * (p & g).sum() / denom) if denom else float("nan"))
    return out


def pick_diverse_cases(val_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n cases spanning severity buckets, biased to coverage over redundancy."""
    bins = pd.cut(
        val_df["cobb_angle_deg"].fillna(-1),
        bins=[-2, 0, 10, 25, 40, 180],
        labels=["normal", "subclinical", "mild", "moderate", "severe"],
    )
    val_df = val_df.assign(_bucket=bins.astype(str))
    print(f"  bucket dist: {val_df['_bucket'].value_counts().to_dict()}")

    targets = {"normal": 1, "mild": 1, "moderate": 2, "severe": 2}
    if n != 6:
        targets = {k: max(1, n // 4) for k in ["normal", "mild", "moderate", "severe"]}
    picks = []
    for b, k in targets.items():
        sub = val_df[val_df["_bucket"] == b]
        if len(sub) > 0:
            picks.append(sub.sample(min(k, len(sub)), random_state=7))
    out = pd.concat(picks) if picks else val_df.sample(n, random_state=7)
    if len(out) < n:
        remaining = val_df[~val_df.index.isin(out.index)]
        if len(remaining):
            out = pd.concat([out, remaining.sample(min(n - len(out), len(remaining)), random_state=7)])
    out = out.sort_values("cobb_angle_deg", na_position="first")
    return out.head(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=str, default="/tmp/inspect_predictions.png")
    ap.add_argument("--tta", choices=["off", "hflip"], default="hflip")
    ap.add_argument("--all", action="store_true", help="all val cases sorted by mc Dice asc")
    ap.add_argument("--rows-per-page", type=int, default=15, help="paginate --all output")
    args = ap.parse_args()

    sentinel = json.loads(SENTINEL.read_text())
    run_dir = REPO / sentinel["folds"][args.fold]["run_dir"]
    print(f"loading fold {args.fold} from {run_dir.name}")
    print(f"  reported val Dice: {sentinel['folds'][args.fold]['best_val_dice']:.4f}")

    cpu_dev = torch.device("cpu")
    predictor = Predictor(run_dir, device=cpu_dev)
    print(f"  device: {predictor.device}")
    print(f"  clahe_mode: {predictor.clahe_mode}  roi_crop_mode: {predictor.roi_crop_mode}")

    splits = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT)
    spec = splits[args.fold]
    full_df = pd.read_csv(CLEAN_INDEX)
    pool = trainable_rows(full_df, min_target_count=14)
    val_df = pool.loc[list(spec.val_idx)].reset_index(drop=True).copy()
    print(f"  fold {args.fold} val cases: {len(val_df)}")

    cmap = vertebra_cmap()

    if args.all:
        print(f"  running inference on ALL {len(val_df)} val cases...")
        cache = []
        for idx, (_, row) in enumerate(val_df.iterrows()):
            out = predictor.predict_from_row(row, tta=args.tta)
            img = out["image"].squeeze().cpu().numpy()
            gt = out["seg"].cpu().numpy().astype(np.int32)
            pred = out["pred"].cpu().numpy().astype(np.int32)
            d_bin = dice_score(pred, gt)
            d_mc = float(np.nanmean(per_class_dice(pred, gt)))
            cache.append({"row": row, "img": img, "gt": gt, "pred": pred,
                          "d_bin": d_bin, "d_mc": d_mc})
            if (idx + 1) % 10 == 0:
                print(f"    {idx+1}/{len(val_df)} done")
        cache.sort(key=lambda c: c["d_mc"])
        picks_iter = [(c["row"], c) for c in cache]
        n_rows = len(cache)
    else:
        picks = pick_diverse_cases(val_df, args.n)
        print(f"  sampled {len(picks)} cases:")
        for _, r in picks.iterrows():
            cobb = r.get("cobb_angle_deg")
            print(f"    {r['category']} {r['patient_id']}  cobb={cobb}")
        n_rows = len(picks)
        picks_iter = [(row, None) for _, row in picks.iterrows()]

    def render_one_row(row, cached, axes_row):
        if cached is None:
            out = predictor.predict_from_row(row, tta=args.tta)
            img = out["image"].squeeze().cpu().numpy()
            gt = out["seg"].cpu().numpy().astype(np.int32)
            pred = out["pred"].cpu().numpy().astype(np.int32)
            d_bin = dice_score(pred, gt)
            d_mc = float(np.nanmean(per_class_dice(pred, gt)))
        else:
            img, gt, pred = cached["img"], cached["gt"], cached["pred"]
            d_bin, d_mc = cached["d_bin"], cached["d_mc"]
        cobb = row.get("cobb_angle_deg")
        title_suffix = f" | Cobb={cobb:.1f}°" if pd.notna(cobb) else " | (normal)"
        axes_row[0].imshow(img, cmap="gray")
        axes_row[0].set_title(f"{row['category']} {row['patient_id']}{title_suffix}")
        axes_row[0].axis("off")
        axes_row[1].imshow(img, cmap="gray")
        axes_row[1].imshow(gt, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        axes_row[1].set_title("GT (17 vertebrae)")
        axes_row[1].axis("off")
        axes_row[2].imshow(img, cmap="gray")
        axes_row[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        axes_row[2].set_title(f"Pred — bin={d_bin:.2f} | mc={d_mc:.2f}")
        axes_row[2].axis("off")
        diff = np.zeros((*img.shape, 3))
        pred_fg = pred > 0
        gt_fg = gt > 0
        diff[gt_fg & pred_fg] = [0, 1, 0]
        diff[pred_fg & ~gt_fg] = [1, 0, 0]
        diff[gt_fg & ~pred_fg] = [0, 0, 1]
        axes_row[3].imshow(img, cmap="gray")
        axes_row[3].imshow(diff, alpha=0.5)
        axes_row[3].set_title("diff (G=ok R=FP B=FN)")
        axes_row[3].axis("off")
        return d_bin, d_mc

    def render_page(items, page_label, out_path):
        rows = len(items)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 3.2 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        ds = [render_one_row(row, cached, axes[i]) for i, (row, cached) in enumerate(items)]
        bin_m = float(np.mean([d[0] for d in ds]))
        mc_m = float(np.mean([d[1] for d in ds]))
        suptitle = (
            f"Fold {args.fold} EncoderUNet (resnet34 + EMA + boundary λ=0.05 + ROI mask) "
            f"— reported val Dice {sentinel['folds'][args.fold]['best_val_dice']:.4f} (mc)"
        )
        if page_label:
            suptitle += f"  |  {page_label}"
        suptitle += f"\nthis page: binary={bin_m:.3f}  |  multi-class={mc_m:.3f}  (n={rows})"
        fig.suptitle(suptitle, fontsize=13, y=1.0)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.name}  ({out_path.stat().st_size/1024:.1f} KB)  bin={bin_m:.3f} mc={mc_m:.3f}")
        return ds

    out_path = Path(args.out)
    if args.all:
        rpp = args.rows_per_page
        pages = [picks_iter[i:i + rpp] for i in range(0, len(picks_iter), rpp)]
        all_d = []
        for p_idx, p_items in enumerate(pages):
            stem = out_path.with_suffix("").name
            p_path = out_path.parent / f"{stem}_page{p_idx+1}of{len(pages)}.png"
            label = f"WORST→BEST page {p_idx+1}/{len(pages)} — rows {p_idx*rpp+1}..{p_idx*rpp+len(p_items)}"
            all_d.extend(render_page(p_items, label, p_path))
        # leaderboard CSV
        rows_csv = []
        for c in cache:
            rows_csv.append({
                "patient_id": c["row"]["patient_id"],
                "category": c["row"]["category"],
                "cobb_deg": c["row"].get("cobb_angle_deg"),
                "binary_dice": c["d_bin"],
                "mc_dice": c["d_mc"],
            })
        csv_path = out_path.parent / f"{out_path.with_suffix('').name}_leaderboard.csv"
        pd.DataFrame(rows_csv).sort_values("mc_dice").to_csv(csv_path, index=False)
        bin_all = float(np.mean([d[0] for d in all_d]))
        mc_all = float(np.mean([d[1] for d in all_d]))
        print(f"\nALL {len(all_d)} cases: binary={bin_all:.3f}  multi-class={mc_all:.3f}")
        print(f"reported fold {args.fold} val Dice (mc): {sentinel['folds'][args.fold]['best_val_dice']:.4f}")
        print(f"leaderboard CSV: {csv_path}")
    else:
        render_page(picks_iter, None, out_path)


if __name__ == "__main__":
    main()
