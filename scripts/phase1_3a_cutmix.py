"""Phase 1.3a — Phase 1.2 cfg + CutMix (single experimental change).

Tests CutMix's marginal contribution on top of the Phase 1.2 D1+ROI cfg
(resnet34 + EMA + boundary lambda=0.05 + CLAHE off + roi_from_mask), which
is the current 5-fold leader (mean Dice 0.6946 +/- 0.0205, single-split
0.6739). The single override vs Phase 1.2 is
``train.cutmix.enabled = True`` (prob 0.5, rect_frac_range [0.1, 0.4]).

Comparison target: Phase 1.2 single-split val Dice = 0.6739
(cfg-hash 32904622770f0be2). A new cfg-hash is expected because cutmix
fields are part of ``_cache_keys``.

Idempotent: exits if the sentinel already exists.
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
    """Same hook contract as scripts/phase1_2_5fold.py — dvc-add each
    fold's run_dir, dvc push, force-add the sentinel + .dvc pointers,
    commit, push, then optionally shutdown the host. See
    ``feedback_save_model_weights.md``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    out_abs = out_path.resolve()
    log.info("post-run hook: dvc add+push checkpoints + git push sentinel; self_stop=%s", self_stop)

    folds = summary.get("folds")
    if folds is None:
        # Single-split case: synthesize a one-row "folds" list.
        folds = [{"run_dir": summary["run_dir"]}] if "run_dir" in summary else []

    new_dvc_files: list[str] = []
    for fold in folds:
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
            f"phase1_3a sentinel + checkpoints: "
            f"dice={summary.get('best_val_dice', summary.get('mean_dice', 'NA')):.4f}"
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
            ["sudo", "shutdown", "-h", "+2", "phase1_3a complete; auto-stop"],
            check=False, timeout=10,
        )
    except Exception as exc:
        log.warning("shutdown call failed: %s — box will stay up", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--out",
        default="experiments/results/phase1_3a_cutmix.json",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-stop", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("single", "5fold"),
        default="single",
        help="single = canonical 80/20 split (fast smoke); 5fold = full CV gate",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — phase 1.3a already complete, exiting", out_path)
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

    # Phase 1.3a's single experimental change vs Phase 1.2.
    cfg["train"].setdefault("cutmix", {})
    cfg["train"]["cutmix"]["enabled"] = True
    cfg["train"]["cutmix"].setdefault("prob", 0.5)
    cfg["train"]["cutmix"].setdefault("rect_frac_range", [0.1, 0.4])

    # Speed knob — not part of cfg-hash.
    cfg["train"]["num_workers"] = 3

    log.info(
        "Phase 1.3a cfg: encoder=%s clahe=%s boundary=%.2f roi_crop=%s "
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
            "gate_threshold_dice": 0.665,
            "thesis_target_dice": 0.78,
        }
        log.info(
            "Phase 1.3a single-split done — best_val_dice=%.4f vs Phase 1.2 ref 0.6739 (delta %+0.4f)",
            result["best_val_dice"], result["best_val_dice"] - 0.6739,
        )
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
            "gate_threshold_dice": 0.665,
            "thesis_target_dice": 0.78,
            "folds": fold_metrics,
        }
        log.info(
            "Phase 1.3a 5-fold mean Dice = %.4f +/- %.4f (vs Phase 1.2 mean 0.6946, delta %+0.4f)",
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
