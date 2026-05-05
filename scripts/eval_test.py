"""Final test evaluation — touches the frozen ``test_holdout.csv`` ONCE.

Run this only after all gates pass. Reports val Dice + Cobb MAE on
the 25-case test slice that was sealed at the start of the project.
Bootstrapped 95 % CI on Dice and Cobb MAE.

Usage:
    python scripts/eval_test.py --run-dir ai/models/checkpoints/encoder_unet/<TIMESTAMP_HASH>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai.evaluation.cobb import cobb_from_segmentation_tangent
from ai.evaluation.seg_metrics import DatasetDiceAccumulator
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES

log = logging.getLogger(__name__)


def _bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(values[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="path to a trained run dir")
    parser.add_argument("--test-csv", default="data/processed/audit_v2_corrected/test_holdout.csv")
    parser.add_argument("--out", default="experiments/results/test_metrics.json")
    parser.add_argument("--tta", default="hflip", choices=["off", "hflip"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    test_df = pd.read_csv(args.test_csv)
    log.warning(
        "TOUCHING TEST HOLDOUT (%d cases). This should happen ONCE per project, "
        "after all gates have passed.",
        len(test_df),
    )

    predictor = Predictor(args.run_dir)
    device = predictor.device
    acc = DatasetDiceAccumulator(num_classes=NUM_SEG_CLASSES, device=device)
    per_image_dice: list[float] = []
    cobb_abs_errs: list[float] = []

    for _, row in test_df.iterrows():
        out = predictor.predict_from_row(row, tta=args.tta)
        with torch.no_grad():
            logits_pred = torch.zeros(
                (1, NUM_SEG_CLASSES, *out["pred"].shape), device=device
            )
            logits_pred.scatter_(1, out["pred"].to(device).long().unsqueeze(0).unsqueeze(0), 1.0)
            target = out["seg"].to(device).long().unsqueeze(0)
            acc.update(logits_pred, target)

            # Per-image Dice for the bootstrap CI.
            from ai.evaluation.seg_metrics import macro_dice_per_image
            d = macro_dice_per_image(logits_pred, target, num_classes=NUM_SEG_CLASSES)
            per_image_dice.append(float(d.item()))

        gt = row.get("cobb_angle_deg")
        if gt is not None and not pd.isna(gt):
            pred_cobb = cobb_from_segmentation_tangent(out["pred"].numpy())
            cobb_abs_errs.append(abs(float(pred_cobb) - float(gt)))

    pooled_dice = acc.compute()
    dice_arr = np.array(per_image_dice)
    cobb_arr = np.array(cobb_abs_errs)

    dice_lo, dice_hi = _bootstrap_ci(dice_arr)
    cobb_lo, cobb_hi = _bootstrap_ci(cobb_arr)

    metrics = {
        "run_dir": args.run_dir,
        "tta": args.tta,
        "n_test_cases": int(len(test_df)),
        "pooled_dice": float(pooled_dice),
        "per_image_dice_mean": float(dice_arr.mean()) if len(dice_arr) else float("nan"),
        "per_image_dice_ci_lo": dice_lo,
        "per_image_dice_ci_hi": dice_hi,
        "cobb_mae_deg": float(cobb_arr.mean()) if len(cobb_arr) else float("nan"),
        "cobb_mae_ci_lo": cobb_lo,
        "cobb_mae_ci_hi": cobb_hi,
        "n_cobb_cases": int(len(cobb_arr)),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("test pooled Dice = %.3f", pooled_dice)
    log.info(
        "test per-image Dice = %.3f (95%% CI: %.3f, %.3f)",
        metrics["per_image_dice_mean"], dice_lo, dice_hi,
    )
    log.info(
        "test Cobb MAE = %.2f° (95%% CI: %.2f°, %.2f°)",
        metrics["cobb_mae_deg"], cobb_lo, cobb_hi,
    )
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
