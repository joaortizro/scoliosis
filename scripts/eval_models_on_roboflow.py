"""Zero-shot generalization eval: how well do v2-only-trained models segment Jorge's 18 hand-labeled Roboflow radiographs?

Three models, same 18 cases:
  - D1 (`20260517_035932_34b773bbd50ff3b6`): trained on v2_corrected_x2 only, NEVER saw roboflow → ZERO-SHOT
  - fold-0 (`20260509_194823_b41714d16d325371`): trained on v2_corrected only (no x2, no roboflow), well-trained to ep 83 with patience=20 → ZERO-SHOT + early-stop-clean
  - D2 (`20260517_050245_1b8fe848ffa7b4fb`): trained on v2_corrected_x2 + the 18 roboflow → MEMORIZATION (sanity check)

The interesting number is **D1 (or fold-0) on roboflow vs. on v2 val**: tells us how much of the +0.045 D2-over-D1 macro lift came from in-distribution data volume vs. fitting an out-of-distribution shift.
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

from ai.evaluation.seg_metrics import confusion_per_class
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES

log = logging.getLogger(__name__)

ID_TO_NAME = {0:"bg",1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8",9:"T9",
              10:"T10",11:"T11",12:"T12",13:"L1",14:"L2",15:"L3",16:"L4",17:"L5"}


def eval_model(run_dir: str, rows: pd.DataFrame) -> dict:
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    device = predictor.device

    tp_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)
    fp_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)
    fn_total = torch.zeros(NUM_SEG_CLASSES, dtype=torch.float64, device=device)

    per_case_macro: list[tuple[str, float]] = []
    per_case_per_class: list[tuple[str, list[float]]] = []

    for _, row in rows.iterrows():
        case_id = f"{row['category']}_{row['patient_id']}"
        out = predictor.predict_from_row(row, tta="off")
        pred = out["pred"].to(device).long().unsqueeze(0)
        target = out["seg"].to(device).long().unsqueeze(0)

        c = confusion_per_class(pred, target, num_classes=NUM_SEG_CLASSES)
        tp_c, fp_c, fn_c = c["tp"].double(), c["fp"].double(), c["fn"].double()
        case_dice = (2 * tp_c) / (2 * tp_c + fp_c + fn_c + 1e-9)
        present = (tp_c + fn_c) > 0
        case_dice_masked = torch.where(present, case_dice, torch.tensor(float("nan"), device=device))

        per_case_per_class.append((case_id, case_dice_masked.cpu().tolist()))
        fg_present = present[1:]
        macro = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")
        per_case_macro.append((case_id, macro))

        tp_total += tp_c
        fp_total += fp_c
        fn_total += fn_c

    pooled_dice = (2 * tp_total) / (2 * tp_total + fp_total + fn_total + 1e-9)
    pooled_per_class = {ID_TO_NAME[i]: float(pooled_dice[i].item()) for i in range(NUM_SEG_CLASSES)}

    return {
        "pooled_per_class": pooled_per_class,
        "pooled_macro_fg": float(pooled_dice[1:].mean().item()),
        "per_case_macro": per_case_macro,
        "per_case_per_class": per_case_per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-run", default="ai/models/checkpoints/encoder_unet/20260517_035932_34b773bbd50ff3b6")
    parser.add_argument("--d2-run", default="ai/models/checkpoints/encoder_unet/20260517_050245_1b8fe848ffa7b4fb")
    parser.add_argument("--f0-run", default="ai/models/checkpoints/encoder_unet/20260509_194823_b41714d16d325371")
    parser.add_argument("--roboflow-index", default="data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv")
    parser.add_argument("--out", default="experiments/results/zero_shot_on_roboflow.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = pd.read_csv(args.roboflow_index)
    log.info("Roboflow cases: %d (categories: %s)", len(rows), rows["category"].value_counts().to_dict())

    models = [
        ("fold0 (v2_corrected only, ep 83, patience=20)", args.f0_run, "zero-shot, well-trained"),
        ("D1    (v2_corrected_x2, ep 43, patience=10)  ", args.d1_run, "zero-shot, under-trained"),
        ("D2    (v2_corrected_x2 + 18 roboflow, ep 71) ", args.d2_run, "MEMORIZATION (saw these)"),
    ]
    results = {}
    for label, run_dir, note in models:
        log.info("Evaluating %s — %s", label.strip(), note)
        results[label.strip()] = eval_model(run_dir, rows)

    # Headline macro
    print()
    print("=" * 88)
    print(f"Macro foreground Dice on 18 hand-labeled Roboflow radiographs ({len(rows)} cases)")
    print("=" * 88)
    print(f"{'model':<55} {'macro_fg':>10} {'note':<25}")
    print("-" * 88)
    for label, _, note in models:
        r = results[label.strip()]
        print(f"{label:<55} {r['pooled_macro_fg']:>10.4f}  {note}")

    # Per-class — compare fold0 (well-trained, zero-shot) vs D2 (memorization)
    print()
    print("=" * 88)
    print("Per-class pooled Dice — fold0 (zero-shot) vs D1 (zero-shot, undertrained) vs D2 (saw these)")
    print("=" * 88)
    print(f"{'class':<6} {'fold0':>10} {'D1':>10} {'D2':>10} {'Δ D2-f0':>12}")
    print("-" * 60)
    f0r = results["fold0 (v2_corrected only, ep 83, patience=20)"]
    d1r = results["D1    (v2_corrected_x2, ep 43, patience=10)"]
    d2r = results["D2    (v2_corrected_x2 + 18 roboflow, ep 71)"]
    for i in range(NUM_SEG_CLASSES):
        cls = ID_TO_NAME[i]
        v_f0 = f0r["pooled_per_class"][cls]
        v_d1 = d1r["pooled_per_class"][cls]
        v_d2 = d2r["pooled_per_class"][cls]
        delta = v_d2 - v_f0
        marker = " !!" if delta > 0.05 else ""
        print(f"{cls:<6} {v_f0:>10.4f} {v_d1:>10.4f} {v_d2:>10.4f} {delta:>+12.4f}{marker}")
    print("-" * 60)
    print(f"{'macro':<6} {f0r['pooled_macro_fg']:>10.4f} {d1r['pooled_macro_fg']:>10.4f} {d2r['pooled_macro_fg']:>10.4f} {d2r['pooled_macro_fg']-f0r['pooled_macro_fg']:>+12.4f}")

    # Comparison context: same models on the v2 canonical val (already on file)
    print()
    print("Context — same models on the v2 canonical 45-case val (from prior eval):")
    print("  D1 macro_fg = 0.6203, D2 macro_fg = 0.6650 (Δ = +0.0447)")
    print()
    print("Read: the OOD shift from v2 → roboflow tells us whether the +0.045 D2-over-D1 lift on v2 val")
    print("      was about adding data volume (D1's zero-shot on roboflow ~= D1's v2 val) or fitting an OOD shift")
    print("      (D1 on roboflow << D1's v2 val).")

    # Per-case spotlight
    print()
    print("=" * 88)
    print("Per-case macro Dice (D1 zero-shot)")
    print("=" * 88)
    d1_macros = dict(d1r["per_case_macro"])
    f0_macros = dict(f0r["per_case_macro"])
    d2_macros = dict(d2r["per_case_macro"])
    print(f"{'case':<15} {'fold0':>10} {'D1':>10} {'D2':>10}")
    for case in sorted(d1_macros.keys()):
        f = f0_macros.get(case, float("nan"))
        d1 = d1_macros[case]
        d2 = d2_macros.get(case, float("nan"))
        print(f"{case:<15} {f:>10.4f} {d1:>10.4f} {d2:>10.4f}")

    # Persist
    summary = {
        "n_cases": int(len(rows)),
        "roboflow_index": args.roboflow_index,
        "models": {
            "fold0_v2corrected_p20_ep83": {
                "run_dir": args.f0_run,
                "pooled_macro_fg": f0r["pooled_macro_fg"],
                "pooled_per_class": f0r["pooled_per_class"],
                "per_case_macro": {k: v for k, v in f0r["per_case_macro"]},
            },
            "d1_v2corrected_x2_p10_ep43": {
                "run_dir": args.d1_run,
                "pooled_macro_fg": d1r["pooled_macro_fg"],
                "pooled_per_class": d1r["pooled_per_class"],
                "per_case_macro": {k: v for k, v in d1r["per_case_macro"]},
            },
            "d2_x2_plus_roboflow_p10_ep71_MEMORIZATION": {
                "run_dir": args.d2_run,
                "pooled_macro_fg": d2r["pooled_macro_fg"],
                "pooled_per_class": d2r["pooled_per_class"],
                "per_case_macro": {k: v for k, v in d2r["per_case_macro"]},
            },
        },
        "v2_canonical_val_context": {
            "d1_macro_fg": 0.6203,
            "d2_macro_fg": 0.6650,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
