"""Partial-FOV experiment trainer (M1a gentle / M1b aggressive).

Re-uses the Phase 1.2 base cfg (resnet34 + EMA + boundary lambda=0.05 +
CLAHE off + ``preprocess.roi_crop = "from_mask"``) and only flips
``train.augment`` to one of the two ``RandomVerticalCrop`` variants
registered in :mod:`ai.training.augmentation`. Plan source:
``2026-05-15_partial_fov_experiment_plan`` (wiki).

Two variants, controlled by ``--variant``:
- ``gentle``     ``augment = v4_vcrop_gentle``      f ∈ [0.5, 1.0]
- ``aggressive`` ``augment = v4_vcrop_aggressive``  f ∈ [0.3, 1.0]

Idempotent: exits if the sentinel JSON for the chosen variant exists.
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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai.training.splits import make_canonical_split, make_cv_folds  # noqa: E402
from ai.training.trainer import run  # noqa: E402

log = logging.getLogger(__name__)


_VARIANT_AUGMENT = {
    "gentle": "v4_vcrop_gentle",
    "aggressive": "v4_vcrop_aggressive",
}


def _post_run_hook(out_path: Path, summary: dict, *, self_stop: bool) -> None:
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
        subprocess.run(["git", "add", *git_paths], cwd=repo_root, check=False)
        msg = (
            f"partial-FOV {summary['variant']} 5-fold sentinel + checkpoints: "
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

    log.info("scheduling shutdown -h +2")
    try:
        subprocess.run(
            ["sudo", "shutdown", "-h", "+2",
             f"partial-FOV {summary['variant']} 5-fold complete; auto-stop"],
            check=False, timeout=10,
        )
    except Exception as exc:
        log.warning("shutdown call failed: %s — box will stay up", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANT_AUGMENT),
        required=True,
        help="gentle = M1a, aggressive = M1b",
    )
    parser.add_argument("--out", default=None,
                        help="Override sentinel path (default: experiments/results/partial_fov_<variant>_<scope>.json)")
    parser.add_argument(
        "--folds",
        type=int,
        choices=[1, 5],
        default=1,
        help="1 = single canonical 80/20 split (~1h, derisk pass); 5 = full CV (~5.5h).",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-stop", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    augment_name = _VARIANT_AUGMENT[args.variant]
    scope = "single" if args.folds == 1 else "5fold"
    out_path = Path(
        args.out
        or f"experiments/results/partial_fov_{args.variant}_{scope}.json"
    )
    if out_path.exists():
        log.info("sentinel %s exists — variant %s already complete, exiting",
                 out_path, args.variant)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # Re-use the Phase 1.2 production cfg exactly. The ONLY delta is the
    # augment name.
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"
    cfg["train"]["augment"] = augment_name
    cfg["train"]["num_workers"] = 3

    if args.folds == 5:
        splits = make_cv_folds(
            clean_index_csv=cfg["data"]["clean_index"],
            test_holdout_csv=cfg["data"]["test_holdout"],
            n_splits=int(cfg["data"].get("cv_folds", 5)),
            seed=int(cfg["data"]["random_seed"]),
        )
    else:
        splits = [make_canonical_split(
            clean_index_csv=cfg["data"]["clean_index"],
            test_holdout_csv=cfg["data"]["test_holdout"],
            val_frac=float(cfg["data"]["val_frac"]),
            seed=int(cfg["data"]["random_seed"]),
        )]

    log.info(
        "partial-FOV %s %s cfg: augment=%s encoder=%s clahe=%s boundary=%.2f "
        "roi_crop=%s batch=%d lr_dec=%g epochs=%d splits=%d",
        args.variant, scope, augment_name,
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
        # ``spec.fold`` is ``-1`` for the canonical 80/20 split; just label
        # it "single" in the log.
        fold_label = f"{spec.fold + 1}/{len(splits)}" if spec.fold >= 0 else "single"
        log.info("=== split %s (split_hash=%s) ===",
                 fold_label, spec.hash())
        result = run(cfg, spec=spec, use_cache=not args.no_cache)
        log.info(
            "fold %d done — best_val_dice=%.4f source=%s time=%.1fs run_dir=%s",
            spec.fold, result["best_val_dice"], result["best_source"],
            result["total_time_sec"], result["run_dir"],
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
        "variant": args.variant,
        "augment": augment_name,
        "scope": scope,
        "mean_dice": float(dices.mean()),
        "std_dice": float(dices.std(ddof=0)),
        "min_dice": float(dices.min()),
        "max_dice": float(dices.max()),
        "n_folds": int(len(splits)),
        "total_time_sec": float(time.time() - t_start),
        "phase1_2_5fold_mean": 0.6946,
        "phase1_2_single_split_ref": 0.6739,
        "folds": fold_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "partial-FOV %s mean Dice = %.4f +/- %.4f (worst %.4f, best %.4f)",
        args.variant, summary["mean_dice"], summary["std_dice"],
        summary["min_dice"], summary["max_dice"],
    )

    _post_run_hook(out_path, summary, self_stop=args.self_stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
