"""Phase 1.3a-lite — same as phase1_3a_cutmix.py but cutmix.prob=0.25.

Phase 1.3a (cutmix.prob=0.5) landed at single-split Dice 0.6657 (Δ−0.008
vs Phase 1.2's 0.6739) — null/slightly negative result. Hypothesis for
the lite variant: prob=0.5 is too aggressive on a 152-case training
set; gentler regularization (every 4th batch ish) might preserve
augmentation diversity without over-corrupting the supervision signal.

Expected landing: 0.68-0.72 5-fold mean Dice. If still null, CutMix is
not the lever for this dataset and we close the Phase 1.3 ladder.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse the full driver — only override the cutmix.prob default.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_3a_cutmix import _post_run_hook  # noqa: E402,F401  (re-exported)

import argparse
import json
import logging
import time

import numpy as np
import yaml

from ai.training.splits import make_cv_folds
from ai.training.trainer import run

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--out",
        default="experiments/results/phase1_3a_cutmix_lite.json",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-stop", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("single", "5fold"),
        default="5fold",
        help="single or 5fold (default: 5fold)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — phase 1.3a-lite already complete, exiting", out_path)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # Phase 1.2 D1+ROI cfg overrides (must match scripts/phase1_2_5fold.py).
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"

    # The single experimental change vs Phase 1.3a (which had prob=0.5).
    cfg["train"].setdefault("cutmix", {})
    cfg["train"]["cutmix"]["enabled"] = True
    cfg["train"]["cutmix"]["prob"] = 0.25
    cfg["train"]["cutmix"].setdefault("rect_frac_range", [0.1, 0.4])

    cfg["train"]["num_workers"] = 3

    log.info(
        "Phase 1.3a-lite cfg: encoder=%s clahe=%s boundary=%.2f roi_crop=%s "
        "cutmix.prob=%.2f cutmix.frac=%s batch=%d lr_dec=%g epochs=%d mode=%s",
        cfg["train"]["encoder_name"],
        cfg["train"]["preprocess"]["clahe_mode"],
        cfg["train"]["loss"]["boundary_lambda"],
        cfg["train"]["preprocess"]["roi_crop"],
        cfg["train"]["cutmix"]["prob"],
        cfg["train"]["cutmix"]["rect_frac_range"],
        cfg["train"]["batch_size"],
        cfg["train"]["lr_dec"],
        cfg["train"]["epochs"],
        args.mode,
    )

    t_start = time.time()

    if args.mode == "single":
        result = run(cfg, use_cache=not args.no_cache)
        summary = {
            "mode": "single",
            "best_val_dice": result["best_val_dice"],
            "best_source": result["best_source"],
            "total_time_sec": result["total_time_sec"],
            "run_dir": result["run_dir"],
            "phase1_2_single_split_reference": 0.6739,
            "phase1_2_5fold_mean_reference": 0.6946,
            "phase1_3a_cutmix_p05_reference": 0.6657,
            "gate_threshold_dice": 0.665,
            "thesis_target_dice": 0.78,
            "user_target_dice": 0.75,
        }
    else:
        splits = make_cv_folds(
            clean_index_csv=cfg["data"]["clean_index"],
            test_holdout_csv=cfg["data"]["test_holdout"],
            n_splits=int(cfg["data"].get("cv_folds", 5)),
            seed=int(cfg["data"]["random_seed"]),
        )
        fold_metrics = []
        for spec in splits:
            log.info("=== fold %d/%d (split_hash=%s) ===", spec.fold + 1, len(splits), spec.hash())
            result = run(cfg, spec=spec, use_cache=not args.no_cache)
            log.info(
                "fold %d done — best_val_dice=%.4f time=%.1fs run_dir=%s",
                spec.fold, result["best_val_dice"], result["total_time_sec"], result["run_dir"],
            )
            fold_metrics.append({
                "fold": spec.fold,
                "split_hash": spec.hash(),
                "best_val_dice": result["best_val_dice"],
                "best_source": result["best_source"],
                "total_time_sec": result["total_time_sec"],
                "run_dir": result["run_dir"],
            })
        dices = np.array([m["best_val_dice"] for m in fold_metrics])
        summary = {
            "mode": "5fold",
            "mean_dice": float(dices.mean()),
            "std_dice": float(dices.std(ddof=0)),
            "min_dice": float(dices.min()),
            "max_dice": float(dices.max()),
            "n_folds": int(len(splits)),
            "total_time_sec": float(time.time() - t_start),
            "phase1_2_5fold_mean_reference": 0.6946,
            "phase1_3a_cutmix_p05_single_split": 0.6657,
            "gate_threshold_dice": 0.665,
            "thesis_target_dice": 0.78,
            "user_target_dice": 0.75,
            "folds": fold_metrics,
        }
        log.info(
            "Phase 1.3a-lite 5-fold mean Dice = %.4f +/- %.4f (vs Phase 1.2 mean 0.6946, delta %+0.4f)",
            summary["mean_dice"], summary["std_dice"],
            summary["mean_dice"] - 0.6946,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_path)

    _post_run_hook(out_path, summary, self_stop=args.self_stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
