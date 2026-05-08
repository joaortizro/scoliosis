"""Phase 1.1 — TXRV ResNet-50 single-split run.

Idempotent: if ``experiments/results/phase1_1_txrv.json`` already exists,
exits early. Otherwise runs through ``ai.training.trainer.run`` which has
its own cfg-hash cache, so a completed run is reused without retraining.

Trainer does NOT checkpoint per epoch — a mid-run shutdown loses
in-flight epoch progress and the run will restart from epoch 0 on the
next launch (cfg-hash cache only catches *completed* runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from ai.training.trainer import run

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--phase0-summary",
        default="experiments/results/phase0_summary.json",
        help="phase 0 summary used to pick clahe + boundary_lambda winner",
    )
    parser.add_argument(
        "--out",
        default="experiments/results/phase1_1_txrv.json",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — phase 1.1 already complete, exiting", out_path)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    with open(args.phase0_summary) as f:
        ph0 = json.load(f)
    best = max(ph0["results"], key=lambda r: r["best_val_dice"])
    log.info(
        "Phase 0 winner: %s dice=%.3f clahe=%s boundary=%.2f",
        best["name"], best["best_val_dice"], best["clahe_mode"], best["boundary_lambda"],
    )

    cfg["train"]["encoder_name"] = "txrv-resnet50"
    cfg["train"]["preprocess"]["clahe_mode"] = best["clahe_mode"]
    cfg["train"]["loss"]["boundary_lambda"] = best["boundary_lambda"]
    cfg["train"]["preprocess"]["normalization"] = "div255"  # txrv normalization is built into the encoder
    cfg["train"]["ema"]["enabled"] = True
    cfg["train"]["batch_size"] = 6  # 7900 XTX has 24 GB; batch 2 underutilizes the GPU
    cfg["train"]["lr_dec"] = 1.5e-3  # linear-scaling rule vs Phase 0 (batch 4 → 6, lr 1e-3 → 1.5e-3)
    cfg["train"]["num_workers"] = 2  # parallel data prep so GPU doesn't wait on the augment pipeline

    result = run(cfg, use_cache=not args.no_cache)
    log.info("Phase 1.1 done — best_val_dice=%.3f run_dir=%s", result["best_val_dice"], result["run_dir"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
