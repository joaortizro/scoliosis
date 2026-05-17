"""Per-class + per-case Dice eval on D1 and D2 checkpoints over the canonical val split.

Used to settle the 2026-05-17 skepticism: are dataset deltas (D0→D1 mask
corrections, D1→D2 +18 roboflow) visible in per-class numbers, or is the
macro Dice signal too coarse to detect them?

Both models evaluate on the SAME 45-case canonical val (D1's split, pinned
by case_id). Output: per-class Dice for 17 vertebrae + per-case Dice
spotlight on the 6 corrected cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from ai.evaluation.seg_metrics import confusion_per_class, macro_dice_per_image
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training.splits import CASE_ID_COL, make_canonical_split, trainable_rows

log = logging.getLogger(__name__)

ID_TO_NAME = {0:"bg",1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8",9:"T9",
              10:"T10",11:"T11",12:"T12",13:"L1",14:"L2",15:"L3",16:"L4",17:"L5"}

FIXED_CASES = ["Normal_23", "Normal_28", "Normal_36", "Normal_52", "Normal_59", "Normal_71"]


def eval_model_on_val(run_dir: str, val_rows: pd.DataFrame) -> dict:
    # Use CPU — local AMD/DirectML has a known bug loading torch checkpoints
    # via map_location to a torch.device object. CPU works for the ~45 forward
    # passes we need; ~30s extra wall vs GPU is fine.
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    device = predictor.device

    # Global per-class TP/FP/FN accumulators
    tp_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)
    fp_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)
    fn_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)

    per_case_macro: list[tuple[str, float]] = []
    per_case_per_class: list[tuple[str, list[float]]] = []  # one row per case

    for _, row in val_rows.iterrows():
        case_id = f"{row['category']}_{row['patient_id']}"
        out = predictor.predict_from_row(row, tta="off")
        pred = out["pred"].to(device).long().unsqueeze(0)
        target = out["seg"].to(device).long().unsqueeze(0)

        # Per-case per-class Dice (presence-aware)
        c = confusion_per_class(pred, target, num_classes=NUM_SEG_CLASSES)
        tp_c, fp_c, fn_c = c["tp"].double(), c["fp"].double(), c["fn"].double()
        case_dice = (2 * tp_c) / (2 * tp_c + fp_c + fn_c + 1e-9)
        # Mask absent classes (no GT and no prediction) as NaN
        present = (tp_c + fn_c) > 0
        case_dice_masked = torch.where(present, case_dice, torch.tensor(float("nan"), device=device))

        per_case_per_class.append((case_id, case_dice_masked.cpu().tolist()))

        # Per-case macro (mean over foreground classes that are present in GT)
        fg_present = present[1:]  # drop bg
        if fg_present.any():
            macro = float(case_dice[1:][fg_present].mean().item())
        else:
            macro = float("nan")
        per_case_macro.append((case_id, macro))

        tp_total += tp_c
        fp_total += fp_c
        fn_total += fn_c

    # Pooled per-class Dice (sum TP, FP, FN over dataset, then compute)
    pooled_dice = (2 * tp_total) / (2 * tp_total + fp_total + fn_total + 1e-9)
    pooled_dice_per_class = {ID_TO_NAME[i]: float(pooled_dice[i].item()) for i in range(NUM_SEG_CLASSES)}

    return {
        "pooled_per_class": pooled_dice_per_class,
        "pooled_macro_fg": float(pooled_dice[1:].mean().item()),
        "per_case_macro": per_case_macro,
        "per_case_per_class": per_case_per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-run", default="ai/models/checkpoints/encoder_unet/20260517_035932_34b773bbd50ff3b6")
    parser.add_argument("--d2-run", default="ai/models/checkpoints/encoder_unet/20260517_050245_1b8fe848ffa7b4fb")
    parser.add_argument("--clean-index", default="data/processed/audit_v2_corrected_x2/clean_index.csv")
    parser.add_argument("--test-csv", default="data/processed/audit_v2_corrected/test_holdout.csv")
    parser.add_argument("--out", default="experiments/results/per_class_d1_d2.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Resolve canonical val split (D1's clean_index — same indices as D0 by prior forensics)
    spec = make_canonical_split(
        clean_index_csv=args.clean_index,
        test_holdout_csv=args.test_csv,
        val_frac=0.2,
        seed=42,
    )
    full = trainable_rows(pd.read_csv(args.clean_index))
    val_rows = full.iloc[list(spec.val_idx)].copy()
    log.info("Canonical val: %d cases", len(val_rows))

    # Eval D1
    log.info("Evaluating D1 model: %s", args.d1_run)
    d1 = eval_model_on_val(args.d1_run, val_rows)

    # Eval D2 — note D2 val pinned by case_id is same set of v2 cases (no roboflow in val)
    log.info("Evaluating D2 model: %s", args.d2_run)
    d2 = eval_model_on_val(args.d2_run, val_rows)

    # Compare
    print()
    print("=" * 80)
    print("Per-class pooled Dice (D1 vs D2, 45-case canonical val)")
    print("=" * 80)
    print(f"{'class':<6} {'D1':>10} {'D2':>10} {'Δ (D2-D1)':>12}")
    print("-" * 40)
    for i in range(NUM_SEG_CLASSES):
        cls = ID_TO_NAME[i]
        v1 = d1["pooled_per_class"][cls]
        v2 = d2["pooled_per_class"][cls]
        dv = v2 - v1
        marker = " !!" if abs(dv) > 0.03 else ""
        print(f"{cls:<6} {v1:>10.4f} {v2:>10.4f} {dv:>+12.4f}{marker}")
    print("-" * 40)
    print(f"{'macro_fg':<6} {d1['pooled_macro_fg']:>10.4f} {d2['pooled_macro_fg']:>10.4f} {d2['pooled_macro_fg']-d1['pooled_macro_fg']:>+12.4f}")

    print()
    print("=" * 80)
    print("Per-case macro Dice on the 6 corrected cases (D1 vs D2)")
    print("=" * 80)
    d1_macros = dict(d1["per_case_macro"])
    d2_macros = dict(d2["per_case_macro"])
    for c in FIXED_CASES:
        if c in d1_macros:
            v1 = d1_macros[c]
            v2 = d2_macros[c]
            split = "(VAL)" if c in d1_macros else "(not in val)"
            print(f"  {c:<15} {split}  D1={v1:.4f}  D2={v2:.4f}  Δ={v2-v1:+.4f}")
        else:
            print(f"  {c:<15} (not in val — only N_23 is)")

    print()
    print("=" * 80)
    print("N_23 per-class Dice (D1 vs D2) — the one corrected case in val (got +L5)")
    print("=" * 80)
    n23_d1 = next((cls for case_id, cls in d1["per_case_per_class"] if case_id == "Normal_23"), None)
    n23_d2 = next((cls for case_id, cls in d2["per_case_per_class"] if case_id == "Normal_23"), None)
    if n23_d1 is not None:
        print(f"{'class':<6} {'D1':>10} {'D2':>10} {'Δ':>12}")
        for i in range(NUM_SEG_CLASSES):
            v1 = n23_d1[i] if i < len(n23_d1) else float("nan")
            v2 = n23_d2[i] if i < len(n23_d2) else float("nan")
            v1s = f"{v1:.4f}" if not np.isnan(v1) else "n/a"
            v2s = f"{v2:.4f}" if not np.isnan(v2) else "n/a"
            ds = f"{v2-v1:+.4f}" if not np.isnan(v1) and not np.isnan(v2) else "—"
            marker = " ← L5 (the added vertebra)" if ID_TO_NAME[i] == "L5" else ""
            print(f"{ID_TO_NAME[i]:<6} {v1s:>10} {v2s:>10} {ds:>12}{marker}")

    # Persist for the wiki
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "d1_run": args.d1_run,
        "d2_run": args.d2_run,
        "val_clean_index": args.clean_index,
        "n_val": int(len(val_rows)),
        "d1_pooled_per_class": d1["pooled_per_class"],
        "d2_pooled_per_class": d2["pooled_per_class"],
        "d1_macro_fg": d1["pooled_macro_fg"],
        "d2_macro_fg": d2["pooled_macro_fg"],
        "n23_d1_per_class": {ID_TO_NAME[i]: (None if np.isnan(n23_d1[i]) else float(n23_d1[i])) for i in range(NUM_SEG_CLASSES)} if n23_d1 else {},
        "n23_d2_per_class": {ID_TO_NAME[i]: (None if np.isnan(n23_d2[i]) else float(n23_d2[i])) for i in range(NUM_SEG_CLASSES)} if n23_d2 else {},
        "per_case_macro_d1": {k: v for k, v in d1["per_case_macro"]},
        "per_case_macro_d2": {k: v for k, v in d2["per_case_macro"]},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
