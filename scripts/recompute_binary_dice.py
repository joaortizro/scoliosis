"""Recompute binary Dice (foreground vs background) for two evaluations
that originally only logged macro multiclass Dice:

1. **OOD zero-shot on ERS-18**: each of the 5 Phase 1.2 fold checkpoints
   evaluated against the 18 ERS-18 cases (none of which were seen during
   training of the v2-only model).
2. **D2 5-fold val**: each of the 5 D2 fold checkpoints evaluated against
   its v2 val split (pinned to v2 case_ids).

Why this script exists: the paper Table II reports binary Dice for the
RB-UNet base model on test/val, but "---" for D2 val and OOD because the
original sentinels (`dataset_ablation_d2_*.json`, `phase1_2_5fold.json`
roboflow zero-shot eval) only logged macro. This recomputes the binary
aggregate so the table is complete.

Sentinel: `experiments/results/binary_dice_recompute.json`

Runs on CPU (~10-30 min total). The AMD DirectML backend has a
torch.load+map_location bug — same workaround as `eval_test_5fold.py`.
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

from ai.inference.predictor import Predictor
from ai.training.splits import (
    CASE_ID_COL,
    make_cv_folds,
    trainable_rows,
)


PHASE12_SENTINEL = REPO_ROOT / "experiments/results/phase1_2_5fold.json"
D2_SENTINEL = REPO_ROOT / "experiments/results/phase1_2_d2_5fold.json"
ERS18_INDEX = REPO_ROOT / "data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv"
OUT_SENTINEL = REPO_ROOT / "experiments/results/binary_dice_recompute.json"

log = logging.getLogger(__name__)


def binary_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Foreground-vs-background Dice from multiclass prediction + target."""
    pred_fg = (pred > 0)
    gt_fg = (target > 0)
    inter = (pred_fg & gt_fg).sum().double()
    union = pred_fg.sum().double() + gt_fg.sum().double()
    return float((2 * inter / (union + 1e-9)).item())


def eval_run_dir_binary(run_dir: str, rows: pd.DataFrame, tta: str) -> tuple[float, list[float]]:
    """Load checkpoint at run_dir, infer on rows, return (mean_binary_dice, per_case_list)."""
    predictor = Predictor(run_dir, device=torch.device("cpu"))
    per_case: list[float] = []
    for _, row in rows.iterrows():
        out = predictor.predict_from_row(row, tta=tta)
        pred = out["pred"].long()
        target = out["seg"].long()
        per_case.append(binary_dice(pred, target))
    return float(np.mean(per_case)), per_case


def eval_ood_zero_shot(ers18: pd.DataFrame) -> dict:
    """Phase 1.2 5-fold checkpoints, zero-shot on ERS-18 (no training overlap)."""
    sentinel = json.loads(PHASE12_SENTINEL.read_text())
    out = {}
    fold_means: list[float] = []
    log.info("OOD: evaluating %d Phase 1.2 fold checkpoints on %d ERS-18 cases",
             len(sentinel["folds"]), len(ers18))
    for f in sentinel["folds"]:
        t0 = time.time()
        mean, per_case = eval_run_dir_binary(f["run_dir"], ers18, tta="off")
        log.info("  fold %d binary_dice=%.4f (n=%d, %.1fs)",
                 f["fold"], mean, len(per_case), time.time() - t0)
        fold_means.append(mean)
        out[f"fold{f['fold']}"] = {
            "run_dir": f["run_dir"],
            "binary_dice_mean": mean,
            "per_case": per_case,
            "n_cases": len(per_case),
        }
    out["aggregate"] = {
        "mean": float(np.mean(fold_means)),
        "std": float(np.std(fold_means, ddof=1)),
        "fold_means": fold_means,
        "n_folds": len(fold_means),
        "eval_set": "ERS-18 (18 cases)",
        "tta": "off",
    }
    return out


def eval_d2_5fold_val() -> dict:
    """D2 5-fold checkpoints, each evaluated on its v2-pinned val split."""
    sentinel = json.loads(D2_SENTINEL.read_text())
    v2_clean_index = REPO_ROOT / sentinel["v2_clean_index"]
    test_holdout = REPO_ROOT / "data/processed/audit_v2_corrected/test_holdout.csv"

    log.info("D2: reproducing v2 5-fold splits from %s (seed=42)", v2_clean_index)
    v2_splits = make_cv_folds(v2_clean_index, test_holdout, n_splits=5, seed=42)
    v2_df = pd.read_csv(v2_clean_index)
    v2_trainable = trainable_rows(v2_df, min_target_count=14)
    # NOTE: trainable_rows preserves the parent index; make_cv_folds returns
    # indices that are positions in the full df (via full.index[mask].to_numpy()).
    # Use .loc[] not .iloc[] because val_idx values are labels.

    out = {}
    fold_means: list[float] = []
    for fold_idx, f in enumerate(sentinel["folds"]):
        spec = v2_splits[fold_idx]
        val_rows = v2_trainable.loc[list(spec.val_idx)]
        # Sanity: split_hash from spec should match the sentinel's fold hash
        computed_hash = spec.hash()
        if computed_hash != f["split_hash"]:
            log.warning("fold %d: split_hash mismatch (computed=%s sentinel=%s) — "
                        "splits diverged. Continuing but flag in output.",
                        fold_idx, computed_hash[:8], f["split_hash"][:8])
        t0 = time.time()
        mean, per_case = eval_run_dir_binary(f["run_dir"], val_rows, tta="hflip")
        log.info("  D2 fold %d binary_dice=%.4f (n=%d, %.1fs)",
                 f["fold"], mean, len(per_case), time.time() - t0)
        fold_means.append(mean)
        out[f"fold{f['fold']}"] = {
            "run_dir": f["run_dir"],
            "binary_dice_mean": mean,
            "per_case": per_case,
            "n_cases": len(per_case),
        }
    out["aggregate"] = {
        "mean": float(np.mean(fold_means)),
        "std": float(np.std(fold_means, ddof=1)),
        "fold_means": fold_means,
        "n_folds": len(fold_means),
        "eval_set": "v2 val per fold (~45 cases each, pinned from Phase 1.2)",
        "tta": "hflip",
    }
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 70)
    log.info("Binary Dice recompute — OOD zero-shot + D2 5-fold val")
    log.info("=" * 70)

    out = {
        "comment": (
            "Binary Dice aggregates recomputed for paper Table II. "
            "Original sentinels only logged macro multiclass."
        ),
    }

    # ---- 1. OOD zero-shot on ERS-18
    ers18 = pd.read_csv(ERS18_INDEX)
    out["ood_zero_shot"] = eval_ood_zero_shot(ers18)
    agg = out["ood_zero_shot"]["aggregate"]
    log.info("OOD zero-shot aggregate: %.4f ± %.4f over %d folds",
             agg["mean"], agg["std"], agg["n_folds"])

    # ---- 2. D2 5-fold val
    out["d2_5fold_val"] = eval_d2_5fold_val()
    if "error" not in out["d2_5fold_val"]:
        agg = out["d2_5fold_val"]["aggregate"]
        log.info("D2 5-fold val aggregate: %.4f ± %.4f over %d folds",
                 agg["mean"], agg["std"], agg["n_folds"])

    # ---- Sentinel
    OUT_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SENTINEL.write_text(json.dumps(out, indent=2))
    log.info("Sentinel written: %s", OUT_SENTINEL)

    # ---- Console summary for copy into paper
    print()
    print("=" * 70)
    print("PAPER TABLE II — BINARY DICE VALUES TO FILL IN")
    print("=" * 70)
    if "ood_zero_shot" in out:
        a = out["ood_zero_shot"]["aggregate"]
        print(f"OOD zero-shot (ERS-18):   {a['mean']:.4f} ± {a['std']:.4f}")
    if "d2_5fold_val" in out and "error" not in out["d2_5fold_val"]:
        a = out["d2_5fold_val"]["aggregate"]
        print(f"D2 5-fold val:            {a['mean']:.4f} ± {a['std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
