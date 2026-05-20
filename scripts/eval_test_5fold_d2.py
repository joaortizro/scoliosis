"""Second touch of the sealed 25-case test holdout with the D2 5-fold checkpoints.

The single-touch convention was honoured by `eval_test_5fold.py` (Phase 1.2 D1
checkpoints, sentinel `phase1_2_5fold_TEST.json`). This script intentionally
breaks that convention to obtain a comparable test number for the D2 variant
(IBIO-SD + ERS-18 trained model). Rationale: the paper headline initially put
D2 5-fold val 0.7065 in a "complementary" position because D2 was never
evaluated on the sealed set; with this second touch, D2 acquires a rigorous
out-of-sample number that can be reported alongside the D1 test result.

Limitation: re-using the sealed test set once it has been touched is not
"clean" out-of-sample evaluation. The first touch (D1 checkpoints) gave us
information about test-set composition that may have implicitly biased
hyper-parameter / model choices for D2. Reviewers should be aware of this
caveat; the paper text discusses it explicitly.

Sentinel: experiments/results/phase1_2_d2_5fold_TEST.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai.evaluation.cobb import cobb_from_segmentation_tangent
from ai.evaluation.seg_metrics import confusion_per_class
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES, remap_to_target_classes

log = logging.getLogger(__name__)
ID_TO_NAME = {0: "bg", 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6",
              7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11", 12: "T12",
              13: "L1", 14: "L2", 15: "L3", 16: "L4", 17: "L5"}


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def predict_logits(predictor: Predictor, row: pd.Series, tta: str = "hflip"):
    out = predictor.predict_from_row(row, tta=tta)
    pred = out["pred"].long()
    one_hot = torch.zeros(NUM_SEG_CLASSES, *pred.shape, dtype=torch.float32)
    one_hot.scatter_(0, pred.unsqueeze(0), 1.0)
    return one_hot, out["seg"].long()


def per_case_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, list[float]]:
    pred_b = pred.long().unsqueeze(0)
    target_b = target.long().unsqueeze(0)
    c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
    tp, fp, fn = c["tp"].double(), c["fp"].double(), c["fn"].double()
    case_dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    present = (tp + fn) > 0
    fg_present = present[1:]
    macro = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")
    pred_fg = (pred > 0); gt_fg = (target > 0)
    inter = (pred_fg & gt_fg).sum().double()
    union = pred_fg.sum().double() + gt_fg.sum().double()
    binary = float((2 * inter / (union + 1e-9)).item())
    per_class = [float(case_dice[i].item()) if present[i] else float("nan") for i in range(NUM_SEG_CLASSES)]
    return binary, macro, per_class


def eval_fold_on_test(run_dir: str, test_df: pd.DataFrame) -> dict:
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    bin_dices, macro_dices, cobb_errs = [], [], []
    p_tp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    p_fp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    p_fn = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    case_predictions = []
    for _, row in test_df.iterrows():
        one_hot, target = predict_logits(predictor, row, tta="hflip")
        pred = one_hot.argmax(dim=0)
        bin_d, macro_d, _ = per_case_metrics(pred, target)
        bin_dices.append(bin_d); macro_dices.append(macro_d)
        pred_b = pred.long().unsqueeze(0); target_b = target.long().unsqueeze(0)
        c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
        p_tp += c["tp"].double(); p_fp += c["fp"].double(); p_fn += c["fn"].double()
        case_predictions.append((row, one_hot))
        gt_cobb = row.get("cobb_angle_deg")
        if gt_cobb is not None and not pd.isna(gt_cobb):
            pred_cobb = cobb_from_segmentation_tangent(pred.numpy())
            cobb_errs.append(abs(float(pred_cobb) - float(gt_cobb)))
    pooled = (2 * p_tp) / (2 * p_tp + p_fp + p_fn + 1e-9)
    bin_arr = np.array(bin_dices); macro_arr = np.array(macro_dices); cobb_arr = np.array(cobb_errs)
    return {
        "binary_dice_mean": float(bin_arr.mean()),
        "binary_dice_ci": bootstrap_ci(bin_arr),
        "macro_mc_dice_mean": float(macro_arr.mean()),
        "macro_mc_dice_ci": bootstrap_ci(macro_arr),
        "macro_mc_dice_pooled": float(pooled[1:].mean().item()),
        "per_class_pooled": {ID_TO_NAME[i]: float(pooled[i].item()) for i in range(NUM_SEG_CLASSES)},
        "cobb_mae_deg_mean": float(cobb_arr.mean()) if len(cobb_arr) else float("nan"),
        "cobb_mae_ci": bootstrap_ci(cobb_arr) if len(cobb_arr) else (float("nan"), float("nan")),
        "n_cobb_cases": int(len(cobb_arr)),
        "case_predictions": case_predictions,
    }


def ensemble_eval(case_preds_per_fold: list, sentinel_first_fold_rundir: str) -> dict:
    """Average one-hot predictions across folds, then argmax, then compute metrics."""
    n_cases = len(case_preds_per_fold[0])
    ens_bin, ens_macro, ens_cobb = [], [], []
    ens_tp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    ens_fp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    ens_fn = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    for i in range(n_cases):
        row = case_preds_per_fold[0][i][0]
        stacked = torch.stack([case_preds_per_fold[fold_i][i][1] for fold_i in range(len(case_preds_per_fold))])
        avg = stacked.mean(dim=0)
        ens_pred = avg.argmax(dim=0)
        # Reconstruct target from GT mask path
        gt_arr = np.array(Image.open(row["multiclass_mask_path"]))
        if gt_arr.ndim == 3:
            gt_arr = gt_arr[..., 0]
        gt_remap = remap_to_target_classes(gt_arr)
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
    pooled = (2 * ens_tp) / (2 * ens_tp + ens_fp + ens_fn + 1e-9)
    bin_arr = np.array(ens_bin); macro_arr = np.array(ens_macro); cobb_arr = np.array(ens_cobb)
    return {
        "binary_dice_mean": float(bin_arr.mean()),
        "binary_dice_ci": bootstrap_ci(bin_arr),
        "macro_mc_dice_mean": float(macro_arr.mean()),
        "macro_mc_dice_ci": bootstrap_ci(macro_arr),
        "macro_mc_dice_pooled": float(pooled[1:].mean().item()),
        "per_class_pooled": {ID_TO_NAME[i]: float(pooled[i].item()) for i in range(NUM_SEG_CLASSES)},
        "cobb_mae_deg_mean": float(cobb_arr.mean()) if len(cobb_arr) else float("nan"),
        "cobb_mae_ci": bootstrap_ci(cobb_arr) if len(cobb_arr) else (float("nan"), float("nan")),
        "n_cobb_cases": int(len(cobb_arr)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path = REPO_ROOT / "experiments/results/phase1_2_d2_5fold_TEST.json"
    if out_path.exists():
        log.warning("D2 test sentinel %s already exists. Aborting (this would be third+ touch).", out_path)
        log.info("contents: %s", out_path.read_text())
        return

    test_df = pd.read_csv(REPO_ROOT / "data/processed/audit_v2_corrected/test_holdout.csv")
    log.warning("SECOND TOUCH of test holdout (%d cases) — D2 checkpoints. "
                "This intentionally breaks the single-touch convention; "
                "see docstring for justification.", len(test_df))

    sentinel = json.load(open(REPO_ROOT / "experiments/results/phase1_2_d2_5fold.json"))
    fold_results = {}
    case_preds_per_fold = []
    for f in sentinel["folds"]:
        fold = f["fold"]
        log.info("=== D2 fold %d/%d on test (%s) ===", fold + 1, len(sentinel["folds"]), f["run_dir"])
        r = eval_fold_on_test(f["run_dir"], test_df)
        case_preds_per_fold.append(r.pop("case_predictions"))
        fold_results[f"fold{fold}"] = r

    log.info("=== D2 ensemble (avg one-hot, then argmax) ===")
    ensemble = ensemble_eval(case_preds_per_fold, sentinel["folds"][0]["run_dir"])

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
            "mean_dice": sentinel.get("mean_dice"),
            "std_dice": sentinel.get("std_dice"),
        },
        "single_touch_note": (
            "D2 test evaluation is a deliberate SECOND TOUCH of the test holdout; "
            "the first touch was Phase 1.2 D1 (see phase1_2_5fold_TEST.json). "
            "Adds a comparable rigorous out-of-sample number for D2 vs D1. "
            "Limitation: information from first touch may have implicitly biased "
            "D2 model selection."
        ),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Sentinel written: %s", out_path)

    # Console summary
    print()
    print("=" * 75)
    print("D2 TEST RESULTS — SECOND TOUCH OF SEALED HOLDOUT")
    print("=" * 75)
    print(f"Per-fold mean macro mc Dice : {summary['per_fold_mean_macro']:.4f} ± {summary['per_fold_std_macro']:.4f}")
    print(f"Per-fold mean binary Dice   : {summary['per_fold_mean_binary']:.4f} ± {summary['per_fold_std_binary']:.4f}")
    print(f"Per-fold mean Cobb MAE (°)  : {summary['per_fold_mean_cobb']:.2f} ± {summary['per_fold_std_cobb']:.2f}")
    print(f"Ensemble macro mc Dice      : {ensemble['macro_mc_dice_mean']:.4f}")
    print(f"Ensemble binary Dice        : {ensemble['binary_dice_mean']:.4f}")
    print(f"Ensemble Cobb MAE (°)       : {ensemble['cobb_mae_deg_mean']:.2f}")
    print("=" * 75)
    print(f"For comparison — D1 5-fold test (first touch):")
    d1_test = json.load(open(REPO_ROOT / "experiments/results/phase1_2_5fold_TEST.json"))
    print(f"  D1 per-fold mean macro    : {d1_test['per_fold_mean_macro']:.4f} ± {d1_test['per_fold_std_macro']:.4f}")
    print(f"  D1 per-fold mean binary   : {d1_test['per_fold_mean_binary']:.4f} ± {d1_test['per_fold_std_binary']:.4f}")
    print(f"  D1 per-fold mean Cobb MAE : {d1_test['per_fold_mean_cobb']:.2f} ± {d1_test['per_fold_std_cobb']:.2f}")
    print(f"  Δ D2 vs D1 (macro)        : {summary['per_fold_mean_macro'] - d1_test['per_fold_mean_macro']:+.4f}")
    print(f"  Δ D2 vs D1 (binary)       : {summary['per_fold_mean_binary'] - d1_test['per_fold_mean_binary']:+.4f}")
    print(f"  Δ D2 vs D1 (Cobb)         : {summary['per_fold_mean_cobb'] - d1_test['per_fold_mean_cobb']:+.2f}°")
    print("=" * 75)


if __name__ == "__main__":
    main()
