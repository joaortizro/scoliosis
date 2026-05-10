"""Phase 1.2 — 5-fold CV gate for the D1 + ROI cfg.

Runs the Phase 1.2 cfg (resnet34 + EMA + boundary lambda=0.05 + CLAHE off
+ ``preprocess.roi_crop = "from_mask"``) across the 5 folds emitted by
:func:`ai.training.splits.make_cv_folds` and reports mean +/- std macro
val Dice. Single-split headline: 0.6739 (run dir cfg-hash
``32904622770f0be2``). Test slice stays sealed.

Idempotent: exits if ``experiments/results/phase1_2_5fold.json`` exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from ai.training.splits import make_cv_folds
from ai.training.trainer import run

log = logging.getLogger(__name__)


def _post_run_hook(out_path: Path, summary: dict, *, self_stop: bool) -> None:
    """Push checkpoints + sentinel to remote so a fresh box can recover them,
    then optionally power off the host (EBS default = stop on shutdown,
    needs no IAM ec2:StopInstances).

    Per ``feedback_save_model_weights.md``, every fold's run_dir under
    ``ai/models/checkpoints/`` is ``dvc add``-ed before the ``dvc push`` so
    the heavy artifacts actually land in S3. Without this step the trainer
    leaves weights only on the ephemeral host and a stop loses them.
    """
    repo_root = Path(__file__).resolve().parent.parent
    out_abs = out_path.resolve()
    log.info("post-run hook: dvc add+push checkpoints + git push sentinel; self_stop=%s", self_stop)

    new_dvc_files: list[str] = []
    for fold in summary.get("folds", []):
        run_dir = fold.get("run_dir")
        if not run_dir:
            continue
        try:
            subprocess.run(
                ["dvc", "add", run_dir],
                cwd=repo_root, check=False, timeout=300,
            )
            new_dvc_files.append(f"{run_dir}.dvc")
        except Exception as exc:
            log.warning("dvc add %s failed: %s", run_dir, exc)

    try:
        subprocess.run(
            ["dvc", "push"] + new_dvc_files,
            cwd=repo_root, check=False, timeout=1800,
        )
    except Exception as exc:
        log.warning("dvc push failed: %s", exc)

    try:
        sentinel_rel = str(out_abs.relative_to(repo_root))
        git_paths = ["-f", sentinel_rel]
        for dvc_file in new_dvc_files:
            git_paths.extend(["-f", dvc_file])
        subprocess.run(
            ["git", "add", *git_paths],
            cwd=repo_root, check=False,
        )
        msg = (
            f"phase1_2 5-fold sentinel + checkpoints: "
            f"mean={summary['mean_dice']:.4f} std={summary['std_dice']:.4f} "
            f"n={summary['n_folds']}"
        )
        subprocess.run(
            ["git", "-c", "user.email=ec2-auto@scoliosis", "-c", "user.name=EC2 auto",
             "commit", "-m", msg],
            cwd=repo_root, check=False,
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=repo_root, check=False, timeout=120,
        )
    except Exception as exc:
        log.warning("git commit/push failed: %s", exc)

    if not self_stop:
        log.info("self_stop disabled — leaving box up")
        return

    log.info("scheduling shutdown -h +2 (gives 2 min for log flush)")
    try:
        subprocess.run(
            ["sudo", "shutdown", "-h", "+2",
             "phase1_2 5-fold complete; auto-stop"],
            check=False, timeout=10,
        )
    except Exception as exc:
        log.warning("shutdown call failed: %s — box will stay up", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out", default="experiments/results/phase1_2_5fold.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--self-stop",
        action="store_true",
        help="After sentinel write, dvc push + git push + sudo shutdown -h +2 the host.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — phase 1.2 5-fold already complete, exiting", out_path)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # Phase 1.2 cfg — must match scripts/phase1_2_d1_roi.py to keep the
    # cfg-hash stable across the single-split (0.6739) and 5-fold runs.
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"

    # Speed knob — not part of cfg-hash.
    cfg["train"]["num_workers"] = 3

    splits = make_cv_folds(
        clean_index_csv=cfg["data"]["clean_index"],
        test_holdout_csv=cfg["data"]["test_holdout"],
        n_splits=int(cfg["data"].get("cv_folds", 5)),
        seed=int(cfg["data"]["random_seed"]),
    )

    log.info(
        "Phase 1.2 5-fold cfg: encoder=%s clahe=%s boundary=%.2f roi_crop=%s "
        "batch=%d lr_dec=%g epochs=%d folds=%d",
        cfg["train"]["encoder_name"],
        cfg["train"]["preprocess"]["clahe_mode"],
        cfg["train"]["loss"]["boundary_lambda"],
        cfg["train"]["preprocess"]["roi_crop"],
        cfg["train"]["batch_size"],
        cfg["train"]["lr_dec"],
        cfg["train"]["epochs"],
        len(splits),
    )

    fold_metrics = []
    t_start = time.time()
    for spec in splits:
        log.info("=== fold %d/%d (split_hash=%s) ===", spec.fold + 1, len(splits), spec.hash())
        result = run(cfg, spec=spec, use_cache=not args.no_cache)
        log.info(
            "fold %d done — best_val_dice=%.4f source=%s time=%.1fs run_dir=%s",
            spec.fold,
            result["best_val_dice"],
            result["best_source"],
            result["total_time_sec"],
            result["run_dir"],
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
        "single_split_reference": 0.6739,
        "gate_threshold_dice": 0.665,
        "thesis_target_dice": 0.78,
        "folds": fold_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "Phase 1.2 5-fold mean Dice = %.4f +/- %.4f  (worst %.4f, best %.4f)",
        summary["mean_dice"],
        summary["std_dice"],
        summary["min_dice"],
        summary["max_dice"],
    )
    log.info(
        "single-split reference 0.6739; gate threshold 0.665; thesis target 0.78. wrote %s",
        out_path,
    )

    _post_run_hook(out_path, summary, self_stop=args.self_stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
