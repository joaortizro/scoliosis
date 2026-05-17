"""Dataset ablation: single-split comparison of v2_corrected vs v2_corrected_x2 vs +roboflow.

Holds Phase 1.2 D1 + ROI cfg frozen; varies ONLY the dataset.
Val membership is identical for all three runs (same v2 case_ids in val) so
the comparison measures what the model learned, not what's in val.

D2 (with roboflow) gets the 18 extra cases added to TRAIN only.

Variants:
  d1 — v2_corrected_x2 (250 cases, 6 mask overrides vs v2_corrected baseline)
  d2 — v2_corrected_x2 + extra_roboflow (268 cases, +18 in train)

D0 (v2_corrected baseline at 0.6739) is already on record from Phase 1.2 D1 + ROI
single split — not re-run here. Note: 0.6739 used patience=20, this script uses
the current params.yaml (patience=10, min_delta=0.001). Predicted noise ≤ ±0.001
per the wiki §2026-05-16 early-stop note.

Usage:
  python scripts/dataset_ablation_single_split.py --variant d1 [--self-stop]
  python scripts/dataset_ablation_single_split.py --variant d2 [--self-stop]
  python scripts/dataset_ablation_single_split.py --variant all  # run both sequentially
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai.training.splits import (  # noqa: E402
    CASE_ID_COL,
    SplitSpec,
    make_canonical_split,
    trainable_rows,
)
from ai.training.trainer import run  # noqa: E402

log = logging.getLogger(__name__)

VARIANT_CLEAN_INDEX = {
    "d1": "data/processed/audit_v2_corrected_x2/clean_index.csv",
    "d2": "data/processed/audit_v2_corrected_x2_plus_roboflow/clean_index.csv",
}

SENTINEL_PATH = {
    "d1": "experiments/results/dataset_ablation_d1_x2.json",
    "d2": "experiments/results/dataset_ablation_d2_x2_plus_roboflow.json",
}


def build_phase1_2_cfg(params_yaml: str) -> dict:
    """Phase 1.2 D1 + ROI cfg (resnet34, clahe=off, boundary=0.05, roi_from_mask, EMA)."""
    with open(params_yaml) as f:
        cfg = yaml.safe_load(f)

    # Phase 1.2 D1 cfg — matches the 0.6739 baseline as closely as the
    # current params.yaml early-stop permits (patience=10, min_delta=0.001).
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"
    cfg["train"]["num_workers"] = 3
    return cfg


def build_split_d2(
    d1_index_csv: str,
    d2_index_csv: str,
    test_holdout_csv: str,
    val_frac: float,
    seed: int,
) -> SplitSpec:
    """Build D2 split: pin val to D1's canonical val cases, push roboflow into train.

    Ensures the val set is identical between D1 and D2 — clean comparison.
    """
    # Get D1's canonical split (v2-only, 250 cases)
    d1_spec = make_canonical_split(
        clean_index_csv=d1_index_csv,
        test_holdout_csv=test_holdout_csv,
        val_frac=val_frac,
        seed=seed,
    )
    d1_full = trainable_rows(pd.read_csv(d1_index_csv))
    d1_val_case_ids = set(d1_full.iloc[list(d1_spec.val_idx)][CASE_ID_COL])
    d1_test_case_ids = set(d1_full.iloc[list(d1_spec.test_idx)][CASE_ID_COL])

    # Now compute D2's index positions matching the SAME case_ids
    d2_full = trainable_rows(pd.read_csv(d2_index_csv))
    val_idx = d2_full.index[d2_full[CASE_ID_COL].isin(d1_val_case_ids)].tolist()
    test_idx = d2_full.index[d2_full[CASE_ID_COL].isin(d1_test_case_ids)].tolist()
    # train = everything not in val or test (this naturally includes the 18 new roboflow cases)
    in_val_or_test = set(val_idx) | set(test_idx)
    train_idx = [i for i in d2_full.index if i not in in_val_or_test]

    n_v2_train = sum(1 for i in train_idx if d2_full.iloc[i][CASE_ID_COL] in
                     set(d1_full[CASE_ID_COL]) - d1_val_case_ids - d1_test_case_ids)
    n_extra = len(train_idx) - n_v2_train
    log.info(
        "D2 split: %d val (pinned from D1), %d test (pinned from D1), "
        "%d train (= %d v2_corrected_x2 train + %d extra_roboflow)",
        len(val_idx), len(test_idx), len(train_idx), n_v2_train, n_extra,
    )

    return SplitSpec(
        fold=-1,
        train_idx=tuple(int(i) for i in train_idx),
        val_idx=tuple(int(i) for i in val_idx),
        test_idx=tuple(int(i) for i in test_idx),
        seed=seed,
    )


def run_variant(variant: str, params_yaml: str, use_cache: bool = True) -> dict[str, Any]:
    cfg = build_phase1_2_cfg(params_yaml)
    cfg["data"]["clean_index"] = VARIANT_CLEAN_INDEX[variant]

    out_path = Path(SENTINEL_PATH[variant])
    if out_path.exists():
        log.info("sentinel %s exists — variant %s already complete, skipping", out_path, variant)
        return {"variant": variant, "skipped": True, "sentinel": str(out_path)}

    log.info(
        "Variant %s cfg: clean_index=%s encoder=%s clahe=%s boundary=%.2f roi_crop=%s",
        variant, cfg["data"]["clean_index"],
        cfg["train"]["encoder_name"], cfg["train"]["preprocess"]["clahe_mode"],
        cfg["train"]["loss"]["boundary_lambda"], cfg["train"]["preprocess"]["roi_crop"],
    )

    spec = None
    if variant == "d2":
        spec = build_split_d2(
            d1_index_csv=VARIANT_CLEAN_INDEX["d1"],
            d2_index_csv=VARIANT_CLEAN_INDEX["d2"],
            test_holdout_csv=cfg["data"]["test_holdout"],
            val_frac=float(cfg["data"]["val_frac"]),
            seed=int(cfg["data"]["random_seed"]),
        )
    # For d1, spec=None → trainer builds the canonical split internally
    # (= same as baseline d0 since case_ids are identical).

    result = run(cfg, spec=spec, use_cache=use_cache)
    log.info("Variant %s done — best_val_dice=%.4f run_dir=%s",
             variant, result["best_val_dice"], result["run_dir"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "variant": variant,
        "clean_index": VARIANT_CLEAN_INDEX[variant],
        "phase1_2_baseline_d0_single_split": 0.6739,
        "phase1_2_baseline_d0_5fold_mean": 0.6946,
        **{k: v for k, v in result.items() if k != "history"},
    }
    out_path.write_text(json.dumps(out_dict, indent=2, default=str))
    log.info("wrote %s", out_path)
    return out_dict


def post_run_hook(run_dirs: list[str], self_stop: bool) -> None:
    """`dvc add` each run_dir → `dvc push` → optional shutdown. NO git push.

    Per ``feedback_save_model_weights.md`` we MUST ``dvc add`` each
    checkpoint dir before ``dvc push`` so weights actually land in S3.
    Without ``dvc add`` the trainer leaves weights only on the ephemeral
    host and a shutdown loses them.

    Git push is INTENTIONALLY skipped — sentinels + dvc pointers are
    recovered from local via SCP after the box stops (cleaner than
    fighting HTTPS creds + 120s timeouts on EC2).
    """
    repo_root = Path(__file__).resolve().parent.parent
    log.info("post-run hook: dvc add (%d runs) + dvc push; NO git push; self_stop=%s",
             len(run_dirs), self_stop)

    new_dvc_files: list[str] = []
    for run_dir in run_dirs:
        if not run_dir:
            continue
        try:
            subprocess.run(
                ["dvc", "add", run_dir],
                cwd=repo_root, check=False, timeout=300,
            )
            new_dvc_files.append(f"{run_dir}.dvc")
            log.info("dvc add %s OK", run_dir)
        except Exception as exc:
            log.warning("dvc add %s failed: %s", run_dir, exc)

    try:
        subprocess.run(
            ["dvc", "push"] + new_dvc_files,
            cwd=repo_root, check=False, timeout=1800,
        )
        log.info("dvc push complete")
    except Exception as exc:
        log.warning("dvc push failed: %s — checkpoints stay on EC2; SCP-pull before shutdown!", exc)

    if not self_stop:
        log.info("self_stop disabled — leaving box up")
        return

    log.info("scheduling shutdown -h +2 (gives 2 min for log flush)")
    try:
        subprocess.run(
            ["sudo", "shutdown", "-h", "+2",
             "dataset ablation complete; auto-stop"],
            check=False, timeout=10,
        )
    except Exception as exc:
        log.warning("shutdown call failed: %s — box will stay up", exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["d1", "d2", "all"], required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-stop", action="store_true",
                        help="After both sentinels, dvc push + git push + sudo shutdown -h +2.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    variants = ["d1", "d2"] if args.variant == "all" else [args.variant]
    run_dirs: list[str] = []
    for v in variants:
        result = run_variant(v, params_yaml=args.params, use_cache=not args.no_cache)
        log.info("variant %s result: %s", v, json.dumps(result, indent=2, default=str)[:500])
        if isinstance(result, dict) and result.get("run_dir"):
            run_dirs.append(result["run_dir"])

    # Comparison summary
    summary = {}
    for v in variants:
        sp = Path(SENTINEL_PATH[v])
        if sp.exists():
            summary[v] = json.loads(sp.read_text()).get("best_val_dice")
    log.info("=== Ablation Summary ===")
    log.info("D0 (v2_corrected) baseline single split: 0.6739 (Phase 1.2 D1+ROI, prior run)")
    for v, val in summary.items():
        log.info("%s (%s): %s", v, VARIANT_CLEAN_INDEX[v], val)

    # Save weights (dvc add + dvc push) and optionally shutdown.
    # NO git push — sentinels + .dvc pointers will be SCP-pulled from local.
    post_run_hook(run_dirs=run_dirs, self_stop=args.self_stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
