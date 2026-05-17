"""Touch the sealed 25-case test holdout with the Phase 1.2 5-fold checkpoints.

Per-fold test metrics + ensemble (avg-logit) test metrics. Reports macro mc Dice,
binary Dice, Cobb MAE, with bootstrap-CI per-image. Designed to run on CPU (the
AMD DirectML local backend has a torch.load+map_location bug we worked around
in the per-class eval).

Sentinel: experiments/results/phase1_2_5fold_TEST.json — TOUCH ONCE.

Single-touch convention: this is the end-of-project test eval, run after all
gates (Phase 1.2 5-fold gate cleared, partial-FOV charter gates cleared,
dataset extension preliminary in). Per the project rule, the test holdout
must be touched only once; this is that touch.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from ai.evaluation.cobb import cobb_from_segmentation_tangent
from ai.evaluation.seg_metrics import confusion_per_class
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES

log = logging.getLogger(__name__)
ID_TO_NAME = {0:"bg",1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8",9:"T9",
              10:"T10",11:"T11",12:"T12",13:"L1",14:"L2",15:"L3",16:"L4",17:"L5"}


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(values[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def predict_logits(predictor: Predictor, row: pd.Series, tta: str = "hflip") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, target) where probs is (C, H, W) float and target is (H, W) long."""
    out = predictor.predict_from_row(row, tta=tta)
    # predictor returns argmax in out['pred']; for ensemble we need a probabilistic-ish surrogate.
    # Use the one-hot encoding of the prediction since the predictor doesn't expose logits directly.
    pred = out["pred"].long()
    one_hot = torch.zeros(NUM_SEG_CLASSES, *pred.shape, dtype=torch.float32)
    one_hot.scatter_(0, pred.unsqueeze(0), 1.0)
    return one_hot, out["seg"].long()


def per_case_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, list[float]]:
    """Return (binary_dice, macro_fg_dice, per_class_dice_list[18])."""
    pred_b = pred.long().unsqueeze(0)
    target_b = target.long().unsqueeze(0)
    c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
    tp, fp, fn = c["tp"].double(), c["fp"].double(), c["fn"].double()
    case_dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    present = (tp + fn) > 0
    fg_present = present[1:]
    macro = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")
    pred_fg = (pred > 0)
    gt_fg = (target > 0)
    inter = (pred_fg & gt_fg).sum().double()
    union = pred_fg.sum().double() + gt_fg.sum().double()
    binary = float((2 * inter / (union + 1e-9)).item())
    per_class = [float(case_dice[i].item()) if present[i] else float("nan") for i in range(NUM_SEG_CLASSES)]
    return binary, macro, per_class


def eval_fold_on_test(run_dir: str, test_df: pd.DataFrame) -> dict:
    """Eval one fold's checkpoint on the test holdout. Returns metrics + per-case predictions
    (one-hot tensors) for the ensemble averaging."""
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    bin_dices, macro_dices = [], []
    cobb_errs = []
    per_class_pooled_tp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    per_class_pooled_fp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    per_class_pooled_fn = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    case_predictions = []  # for ensemble

    for _, row in test_df.iterrows():
        one_hot, target = predict_logits(predictor, row, tta="hflip")
        pred = one_hot.argmax(dim=0)
        bin_d, macro_d, _ = per_case_metrics(pred, target)
        bin_dices.append(bin_d); macro_dices.append(macro_d)

        # pool TP/FP/FN
        pred_b = pred.long().unsqueeze(0)
        target_b = target.long().unsqueeze(0)
        c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
        per_class_pooled_tp += c["tp"].double()
        per_class_pooled_fp += c["fp"].double()
        per_class_pooled_fn += c["fn"].double()

        case_predictions.append((row, one_hot))

        gt_cobb = row.get("cobb_angle_deg")
        if gt_cobb is not None and not pd.isna(gt_cobb):
            pred_cobb = cobb_from_segmentation_tangent(pred.numpy())
            cobb_errs.append(abs(float(pred_cobb) - float(gt_cobb)))

    pooled_dice = (2 * per_class_pooled_tp) / (2 * per_class_pooled_tp + per_class_pooled_fp + per_class_pooled_fn + 1e-9)
    macro_pooled = float(pooled_dice[1:].mean().item())

    bin_arr = np.array(bin_dices); macro_arr = np.array(macro_dices); cobb_arr = np.array(cobb_errs)
    bin_lo, bin_hi = bootstrap_ci(bin_arr)
    macro_lo, macro_hi = bootstrap_ci(macro_arr)
    cobb_lo, cobb_hi = bootstrap_ci(cobb_arr) if len(cobb_arr) else (float("nan"), float("nan"))

    return {
        "binary_dice_mean": float(bin_arr.mean()),
        "binary_dice_ci": (bin_lo, bin_hi),
        "macro_mc_dice_mean": float(macro_arr.mean()),
        "macro_mc_dice_ci": (macro_lo, macro_hi),
        "macro_mc_dice_pooled": macro_pooled,
        "per_class_pooled": {ID_TO_NAME[i]: float(pooled_dice[i].item()) for i in range(NUM_SEG_CLASSES)},
        "cobb_mae_deg_mean": float(cobb_arr.mean()) if len(cobb_arr) else float("nan"),
        "cobb_mae_ci": (cobb_lo, cobb_hi),
        "n_cobb_cases": int(len(cobb_arr)),
        "case_predictions": case_predictions,  # for ensemble — popped before serialize
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path = Path("experiments/results/phase1_2_5fold_TEST.json")
    if out_path.exists():
        log.warning("sentinel %s already exists — test was already touched. Aborting.", out_path)
        log.info("contents: %s", out_path.read_text())
        return

    test_df = pd.read_csv("data/processed/audit_v2_corrected/test_holdout.csv")
    log.warning("TOUCHING TEST HOLDOUT (%d cases). End-of-project single-touch convention.", len(test_df))

    sentinel = json.load(open("experiments/results/phase1_2_5fold.json"))
    fold_results = {}
    case_preds_per_fold = []
    for f in sentinel["folds"]:
        fold = f["fold"]
        log.info("=== fold %d/%d on test (%s) ===", fold + 1, len(sentinel["folds"]), f["run_dir"])
        r = eval_fold_on_test(f["run_dir"], test_df)
        case_preds_per_fold.append(r.pop("case_predictions"))
        fold_results[f"fold{fold}"] = r

    # Ensemble: average one-hot tensors across folds per case, then argmax
    log.info("=== ensemble (avg one-hot, then argmax) ===")
    n_cases = len(case_preds_per_fold[0])
    ens_bin, ens_macro, ens_cobb = [], [], []
    ens_tp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    ens_fp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    ens_fn = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    for i in range(n_cases):
        row = case_preds_per_fold[0][i][0]
        stacked = torch.stack([case_preds_per_fold[fold_i][i][1] for fold_i in range(5)])
        avg = stacked.mean(dim=0)
        ens_pred = avg.argmax(dim=0)
        # GT
        from PIL import Image
        import numpy as np
        # Easier: re-run predictor once to get target (same predictor, any fold works for shape)
        _, target = predict_logits(Predictor(sentinel["folds"][0]["run_dir"], device=torch.device("cpu")), row, tta="off") if i == -999 else (None, case_preds_per_fold[0][i][1].argmax(dim=0))
        # Reconstruct target by reading GT mask from row
        from ai.preprocessing.segmentation import remap_to_target_classes
        gt_arr = np.array(Image.open(row["multiclass_mask_path"]))
        if gt_arr.ndim == 3: gt_arr = gt_arr[..., 0]
        gt_remap = remap_to_target_classes(gt_arr)
        # Resize to match prediction shape via nearest
        import cv2
        h, w = ens_pred.shape
        gt_resized = cv2.resize(gt_remap.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        target = torch.from_numpy(gt_resized).long()

        bin_d, macro_d, _ = per_case_metrics(ens_pred, target)
        ens_bin.append(bin_d); ens_macro.append(macro_d)
        pred_b = ens_pred.long().unsqueeze(0); target_b = target.long().unsqueeze(0)
        c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
        ens_tp += c["tp"].double(); ens_fp += c["fp"].double(); ens_fn += c["fn"].double()
        gt_cobb = row.get("cobb_angle_deg")
        if gt_cobb is not None and not pd.isna(gt_cobb):
            pred_cobb = cobb_from_segmentation_tangent(ens_pred.numpy())
            ens_cobb.append(abs(float(pred_cobb) - float(gt_cobb)))

    ens_pooled = (2 * ens_tp) / (2 * ens_tp + ens_fp + ens_fn + 1e-9)
    ens_bin_arr = np.array(ens_bin); ens_macro_arr = np.array(ens_macro); ens_cobb_arr = np.array(ens_cobb)
    ens_bin_lo, ens_bin_hi = bootstrap_ci(ens_bin_arr)
    ens_macro_lo, ens_macro_hi = bootstrap_ci(ens_macro_arr)
    ens_cobb_lo, ens_cobb_hi = bootstrap_ci(ens_cobb_arr) if len(ens_cobb_arr) else (float("nan"), float("nan"))

    ensemble = {
        "binary_dice_mean": float(ens_bin_arr.mean()),
        "binary_dice_ci": (ens_bin_lo, ens_bin_hi),
        "macro_mc_dice_mean": float(ens_macro_arr.mean()),
        "macro_mc_dice_ci": (ens_macro_lo, ens_macro_hi),
        "macro_mc_dice_pooled": float(ens_pooled[1:].mean().item()),
        "per_class_pooled": {ID_TO_NAME[i]: float(ens_pooled[i].item()) for i in range(NUM_SEG_CLASSES)},
        "cobb_mae_deg_mean": float(ens_cobb_arr.mean()) if len(ens_cobb_arr) else float("nan"),
        "cobb_mae_ci": (ens_cobb_lo, ens_cobb_hi),
        "n_cobb_cases": int(len(ens_cobb_arr)),
    }

    # Per-fold aggregate
    macros = [fold_results[f]["macro_mc_dice_mean"] for f in fold_results]
    bins = [fold_results[f]["binary_dice_mean"] for f in fold_results]
    cobbs = [fold_results[f]["cobb_mae_deg_mean"] for f in fold_results if not np.isnan(fold_results[f]["cobb_mae_deg_mean"])]

    summary = {
        "n_test_cases": int(len(test_df)),
        "tta": "hflip",
        "per_fold": fold_results,
        "per_fold_mean_macro": float(np.mean(macros)),
        "per_fold_std_macro": float(np.std(macros)),
        "per_fold_mean_binary": float(np.mean(bins)),
        "per_fold_std_binary": float(np.std(bins)),
        "per_fold_mean_cobb": float(np.mean(cobbs)) if cobbs else float("nan"),
        "per_fold_std_cobb": float(np.std(cobbs)) if cobbs else float("nan"),
        "ensemble": ensemble,
        "val_5fold_reference": {
            "mean_dice": sentinel["mean_dice"],
            "std_dice": sentinel["std_dice"],
        },
    }

    print()
    print("=" * 90)
    print(f"Test holdout ({len(test_df)} cases) — Phase 1.2 5-fold ensemble + per-fold")
    print("=" * 90)
    print(f"{'metric':<22} {'per-fold mean±std':<22} {'ensemble':<22} {'val 5-fold (ref)':<22}")
    print(f"{'macro mc Dice':<22} {np.mean(macros):.4f} ± {np.std(macros):.4f}      {ensemble['macro_mc_dice_mean']:.4f} [{ens_macro_lo:.4f}, {ens_macro_hi:.4f}]  {sentinel['mean_dice']:.4f} ± {sentinel['std_dice']:.4f}")
    print(f"{'binary Dice':<22} {np.mean(bins):.4f} ± {np.std(bins):.4f}      {ensemble['binary_dice_mean']:.4f} [{ens_bin_lo:.4f}, {ens_bin_hi:.4f}]  (n/a)")
    if cobbs:
        print(f"{'Cobb MAE (deg)':<22} {np.mean(cobbs):.2f} ± {np.std(cobbs):.2f}      {ensemble['cobb_mae_deg_mean']:.2f} [{ens_cobb_lo:.2f}, {ens_cobb_hi:.2f}]  (n/a)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
