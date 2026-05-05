"""5-fold CV runner — Phase 1 gate driver.

Runs the full Phase 1 stack (TXRV + ROI crop + EMA + …) across the 5
folds emitted by :func:`ai.training.splits.make_cv_folds`. Reports
mean ± std macro Dice and per-fold Cobb MAE. Test slice is sealed.

Usage:
    python scripts/cv5_train.py [--params params.yaml] [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from ai.training.splits import make_cv_folds
from ai.training.trainer import run

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override params.train.epochs")
    parser.add_argument("--out", default="experiments/results/cv5_metrics.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.params) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    splits = make_cv_folds(
        clean_index_csv=cfg["data"]["clean_index"],
        test_holdout_csv=cfg["data"]["test_holdout"],
        n_splits=int(cfg["data"].get("cv_folds", 5)),
        seed=int(cfg["data"]["random_seed"]),
    )

    fold_metrics = []
    t_start = time.time()
    for spec in splits:
        log.info("=== fold %d/%d (hash=%s) ===", spec.fold + 1, len(splits), spec.hash())
        result = run(cfg, spec=spec, use_cache=not args.no_cache)
        log.info(
            "fold %d done — best_val_dice=%.3f  source=%s  time=%.1fs",
            spec.fold, result["best_val_dice"], result["best_source"], result["total_time_sec"],
        )
        fold_metrics.append(
            {
                "fold": spec.fold,
                "split_hash": spec.hash(),
                "best_val_dice": result["best_val_dice"],
                "best_source": result["best_source"],
                "total_time_sec": result["total_time_sec"],
                "run_dir": result["run_dir"],
            }
        )

    dices = np.array([m["best_val_dice"] for m in fold_metrics])
    summary = {
        "mean_dice": float(dices.mean()),
        "std_dice": float(dices.std(ddof=0)),
        "min_dice": float(dices.min()),
        "max_dice": float(dices.max()),
        "n_folds": int(len(splits)),
        "total_time_sec": float(time.time() - t_start),
        "folds": fold_metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "5-fold mean Dice = %.3f ± %.3f  (worst %.3f, best %.3f)",
        summary["mean_dice"], summary["std_dice"],
        summary["min_dice"], summary["max_dice"],
    )
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
