"""All 5 Phase 1.2 fold checkpoints, zero-shot on the 18 roboflow cases.

Each fold was trained on a different 200-case train split of v2_corrected.
None of them saw any of the 18 roboflow cases. This gives us a robust
mean±std for the "v2-only zero-shot generalization to clean-GT data" number.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from ai.evaluation.seg_metrics import confusion_per_class
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES

log = logging.getLogger(__name__)
ID_TO_NAME = {0:"bg",1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8",9:"T9",
              10:"T10",11:"T11",12:"T12",13:"L1",14:"L2",15:"L3",16:"L4",17:"L5"}


def eval_model(run_dir: str, rows: pd.DataFrame) -> dict:
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    tp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    fp = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    fn = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64)
    per_case_macro = {}
    for _, row in rows.iterrows():
        case_id = f"{row['category']}_{row['patient_id']}"
        out = predictor.predict_from_row(row, tta="off")
        pred = out["pred"].long().unsqueeze(0)
        target = out["seg"].long().unsqueeze(0)
        c = confusion_per_class(pred, target, num_classes=NUM_SEG_CLASSES)
        tp_c, fp_c, fn_c = c["tp"].double(), c["fp"].double(), c["fn"].double()
        case_dice = (2 * tp_c) / (2 * tp_c + fp_c + fn_c + 1e-9)
        present = (tp_c + fn_c) > 0
        fg_present = present[1:]
        per_case_macro[case_id] = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")
        tp += tp_c; fp += fp_c; fn += fn_c
    pooled = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return {
        "pooled_per_class": {ID_TO_NAME[i]: float(pooled[i].item()) for i in range(NUM_SEG_CLASSES)},
        "pooled_macro_fg": float(pooled[1:].mean().item()),
        "per_case_macro": per_case_macro,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sentinel = json.load(open("experiments/results/phase1_2_5fold.json"))
    folds = sentinel["folds"]

    rows = pd.read_csv("data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv")
    log.info("Evaluating %d folds on %d roboflow cases", len(folds), len(rows))

    results = {}
    for f in folds:
        fold = f["fold"]
        log.info("fold %d → %s (best_val_dice=%.4f on v2 fold-%d val)", fold, f["run_dir"], f["best_val_dice"], fold)
        results[f"fold{fold}"] = {
            "run_dir": f["run_dir"],
            "v2_fold_val_dice": f["best_val_dice"],
            **eval_model(f["run_dir"], rows),
        }

    macros = [r["pooled_macro_fg"] for r in results.values()]
    print()
    print("=" * 75)
    print(f"5-fold zero-shot on 18 roboflow cases (none saw any of these in training)")
    print("=" * 75)
    print(f"{'fold':<8} {'v2 val Dice':>13} {'roboflow zero-shot':>22} {'Δ (rf − v2)':>14}")
    print("-" * 70)
    for fold_key in sorted(results):
        r = results[fold_key]
        v2 = r["v2_fold_val_dice"]
        rf = r["pooled_macro_fg"]
        print(f"{fold_key:<8} {v2:>13.4f} {rf:>22.4f} {rf - v2:>+14.4f}")
    print("-" * 70)
    print(f"{'mean':<8} {sentinel['mean_dice']:>13.4f} {np.mean(macros):>22.4f} {np.mean(macros) - sentinel['mean_dice']:>+14.4f}")
    print(f"{'std':<8} {sentinel['std_dice']:>13.4f} {np.std(macros):>22.4f}")

    # Per-class mean ± std across folds (zero-shot on roboflow)
    print()
    print("Per-class pooled Dice on 18 roboflow cases — mean ± std across 5 folds:")
    print(f"{'class':<6} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
    for i in range(NUM_SEG_CLASSES):
        cls = ID_TO_NAME[i]
        vals = [r["pooled_per_class"][cls] for r in results.values()]
        print(f"{cls:<6} {np.mean(vals):>9.4f} {np.std(vals):>9.4f} {min(vals):>9.4f} {max(vals):>9.4f}")

    # Per-case mean ± std
    print()
    print("Per-case mean ± std across 5 folds (zero-shot):")
    case_means = {}
    for case in sorted(next(iter(results.values()))["per_case_macro"]):
        vals = [r["per_case_macro"][case] for r in results.values()]
        case_means[case] = (float(np.mean(vals)), float(np.std(vals)), float(min(vals)), float(max(vals)))
        m, s, lo, hi = case_means[case]
        marker = " ← OUTLIER" if m < 0.5 else ""
        print(f"  {case:<15} mean={m:.4f}  std={s:.4f}  range=[{lo:.4f}, {hi:.4f}]{marker}")

    summary = {
        "n_folds": len(folds),
        "n_roboflow_cases": int(len(rows)),
        "fold_results": {k: {kk: vv for kk, vv in v.items() if kk != "per_case_per_class"} for k, v in results.items()},
        "zero_shot_macro_mean": float(np.mean(macros)),
        "zero_shot_macro_std": float(np.std(macros)),
        "v2_5fold_mean_for_context": sentinel["mean_dice"],
        "v2_5fold_std_for_context": sentinel["std_dice"],
        "per_case_mean_across_folds": {k: v[0] for k, v in case_means.items()},
    }
    Path("experiments/results/zero_shot_5fold_on_roboflow.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote experiments/results/zero_shot_5fold_on_roboflow.json")


if __name__ == "__main__":
    main()
