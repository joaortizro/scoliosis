"""Phase 1.2 D2 — 5-fold CV on D1 + extra_roboflow, val pinned to v2 case_ids per fold.

Apples-to-apples comparison with [[2026-05-10_phase1_2_5fold_done]] (Phase 1.2
5-fold on v2_corrected, mean 0.6946 +/- 0.0205). Per fold:

- val = same v2 case_ids as Phase 1.2 fold k val
- train = Phase 1.2 fold k train ∪ all 18 extra_roboflow cases
- test = same frozen 25-case holdout

Cfg overrides early_stop to patience=20, min_delta=0.0 to match the
Phase 1.2 5-fold convergence regime (the current params.yaml patience=10
was set for partial-FOV runs and under-trains the D1+ROI cfg by ~30 epochs
— see [[2026-05-17_dataset_ablation_d1_d2#Forensics]]).

Sentinel: experiments/results/phase1_2_d2_5fold.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from ai.training.splits import (
    CASE_ID_COL,
    SplitSpec,
    make_cv_folds,
    trainable_rows,
)
from ai.training.trainer import run

log = logging.getLogger(__name__)


V2_CLEAN_INDEX = "data/processed/audit_v2_corrected_x2/clean_index.csv"
D2_CLEAN_INDEX = "data/processed/audit_v2_corrected_x2_plus_roboflow/clean_index.csv"
TEST_HOLDOUT = "data/processed/audit_v2_corrected/test_holdout.csv"


def build_d2_fold_split(v2_spec: SplitSpec, v2_df: pd.DataFrame, d2_df: pd.DataFrame, seed: int) -> SplitSpec:
    """Map a v2 SplitSpec's case_ids to indices in D2's merged clean_index.

    Result: same val + test case_ids as v2 fold k; train includes all v2 train cases
    plus the 18 extra_roboflow cases (which are disjoint from v2 by patient_id).
    """
    v2_val_ids = set(v2_df.iloc[list(v2_spec.val_idx)][CASE_ID_COL])
    v2_test_ids = set(v2_df.iloc[list(v2_spec.test_idx)][CASE_ID_COL])

    val_idx = d2_df.index[d2_df[CASE_ID_COL].isin(v2_val_ids)].tolist()
    test_idx = d2_df.index[d2_df[CASE_ID_COL].isin(v2_test_ids)].tolist()
    in_val_or_test = set(val_idx) | set(test_idx)
    train_idx = [i for i in d2_df.index if i not in in_val_or_test]

    n_v2_train = sum(1 for i in train_idx if d2_df.iloc[i][CASE_ID_COL] in set(v2_df[CASE_ID_COL]))
    n_extra = len(train_idx) - n_v2_train
    log.info(
        "fold %d split: %d val (pinned from v2), %d test, %d train (= %d v2 + %d roboflow)",
        v2_spec.fold, len(val_idx), len(test_idx), len(train_idx), n_v2_train, n_extra,
    )
    return SplitSpec(
        fold=v2_spec.fold,
        train_idx=tuple(int(i) for i in train_idx),
        val_idx=tuple(int(i) for i in val_idx),
        test_idx=tuple(int(i) for i in test_idx),
        seed=seed,
    )


def _post_run_hook(out_path: Path, summary: dict, *, self_stop: bool) -> None:
    """dvc add+push checkpoints; NO git push (per dataset-ablation flow);
    optionally shutdown the host. Sentinels SCP-pulled from local on wake-up.
    """
    repo_root = Path(__file__).resolve().parent.parent
    log.info("post-run hook: dvc add+push %d checkpoints; NO git push; self_stop=%s",
             len(summary.get("folds", [])), self_stop)

    new_dvc_files: list[str] = []
    for fold in summary.get("folds", []):
        run_dir = fold.get("run_dir")
        if not run_dir:
            continue
        try:
            subprocess.run(["dvc", "add", run_dir], cwd=repo_root, check=False, timeout=300)
            new_dvc_files.append(f"{run_dir}.dvc")
            log.info("dvc add %s OK", run_dir)
        except Exception as exc:
            log.warning("dvc add %s failed: %s", run_dir, exc)
    try:
        subprocess.run(["dvc", "push"] + new_dvc_files, cwd=repo_root, check=False, timeout=1800)
        log.info("dvc push complete")
    except Exception as exc:
        log.warning("dvc push failed: %s — checkpoints stay on EC2", exc)

    if not self_stop:
        log.info("self_stop disabled — leaving box up")
        return

    log.info("scheduling shutdown -h +2 (gives 2 min for log flush)")
    try:
        subprocess.run(
            ["sudo", "shutdown", "-h", "+2", "phase1_2 D2 5-fold complete; auto-stop"],
            check=False, timeout=10,
        )
    except Exception as exc:
        log.warning("shutdown failed: %s — box stays up", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out", default="experiments/results/phase1_2_d2_5fold.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-stop", action="store_true",
                        help="After sentinel, dvc add+push + sudo shutdown -h +2.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — already complete, exiting", out_path)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # Phase 1.2 D1 + ROI cfg — same as the prior 5-fold runner
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"
    cfg["train"]["num_workers"] = 3

    # **Critical override**: revert early-stop to match Phase 1.2 5-fold convergence regime.
    # The 2026-05-16 change (patience 20->10, min_delta 0->0.001) under-trains
    # D1+ROI by ~30 epochs (see Forensics in [[2026-05-17_dataset_ablation_d1_d2]]).
    cfg["train"]["early_stop"]["patience"] = 20
    cfg["train"]["early_stop"]["min_delta"] = 0.0

    # Point cfg.data.clean_index at D2's merged variant so the run_dir cfg-hash
    # reflects the actual training data
    cfg["data"]["clean_index"] = D2_CLEAN_INDEX

    # Build the v2 5-fold splits (same canonical splits as Phase 1.2 5-fold)
    v2_splits = make_cv_folds(
        clean_index_csv=V2_CLEAN_INDEX,
        test_holdout_csv=TEST_HOLDOUT,
        n_splits=int(cfg["data"].get("cv_folds", 5)),
        seed=int(cfg["data"]["random_seed"]),
    )

    # Load D2's merged clean_index for the per-fold case_id-pinning
    v2_df = trainable_rows(pd.read_csv(V2_CLEAN_INDEX))
    d2_df = trainable_rows(pd.read_csv(D2_CLEAN_INDEX))

    log.info(
        "D2 5-fold cfg: encoder=%s clahe=%s boundary=%.2f roi_crop=%s "
        "batch=%d lr_dec=%g epochs=%d patience=%d folds=%d",
        cfg["train"]["encoder_name"], cfg["train"]["preprocess"]["clahe_mode"],
        cfg["train"]["loss"]["boundary_lambda"], cfg["train"]["preprocess"]["roi_crop"],
        cfg["train"]["batch_size"], cfg["train"]["lr_dec"],
        cfg["train"]["epochs"], cfg["train"]["early_stop"]["patience"], len(v2_splits),
    )

    fold_metrics: list[dict] = []
    t_start = time.time()
    for v2_spec in v2_splits:
        d2_spec = build_d2_fold_split(v2_spec, v2_df, d2_df, seed=int(cfg["data"]["random_seed"]))
        log.info("=== fold %d/%d (D2-merged, val pinned to v2 case_ids) ===",
                 d2_spec.fold + 1, len(v2_splits))
        result = run(cfg, spec=d2_spec, use_cache=not args.no_cache)
        log.info(
            "fold %d done — best_val_dice=%.4f source=%s time=%.1fs run_dir=%s",
            d2_spec.fold, result["best_val_dice"], result["best_source"],
            result["total_time_sec"], result["run_dir"],
        )
        fold_metrics.append({
            "fold": d2_spec.fold,
            "split_hash": d2_spec.hash() if hasattr(d2_spec, "hash") else "n/a",
            "best_val_dice": result["best_val_dice"],
            "best_source": result["best_source"],
            "total_time_sec": result["total_time_sec"],
            "run_dir": result["run_dir"],
            "n_train": len(d2_spec.train_idx),
            "n_val": len(d2_spec.val_idx),
        })

    dices = np.array([m["best_val_dice"] for m in fold_metrics])
    summary = {
        "mean_dice": float(dices.mean()),
        "std_dice": float(dices.std(ddof=0)),
        "min_dice": float(dices.min()),
        "max_dice": float(dices.max()),
        "n_folds": int(len(v2_splits)),
        "total_time_sec": float(time.time() - t_start),
        "phase1_2_5fold_reference_mean": 0.6946,
        "phase1_2_5fold_reference_std": 0.0205,
        "v2_clean_index": V2_CLEAN_INDEX,
        "d2_clean_index": D2_CLEAN_INDEX,
        "patience": cfg["train"]["early_stop"]["patience"],
        "folds": fold_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "D2 5-fold mean Dice = %.4f +/- %.4f  (worst %.4f, best %.4f)",
        summary["mean_dice"], summary["std_dice"], summary["min_dice"], summary["max_dice"],
    )
    log.info("Phase 1.2 5-fold reference: 0.6946 +/- 0.0205. wrote %s", out_path)

    _post_run_hook(out_path, summary, self_stop=args.self_stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
