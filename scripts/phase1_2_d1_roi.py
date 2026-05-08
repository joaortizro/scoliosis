"""Phase 1.2 — ResNet-34 D1 winner + training-time ROI mask crop.

Tests ROI-crop's marginal contribution on top of the Phase 0 D1 winner
(EMA on, CLAHE off, boundary lambda=0.05, ResNet-34 ImageNet pretrain).
The single override vs D1 is ``preprocess.roi_crop = "from_mask"``.
Comparison target: D1 single-split val Dice = 0.655.

``num_workers=2`` is also set, but ``num_workers`` is not part of the
cfg-hash, so it does not invalidate any cached run — purely a runtime
speed knob.

Idempotent: exits if ``experiments/results/phase1_2_d1_roi.json`` exists.
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
        "--out",
        default="experiments/results/phase1_2_d1_roi.json",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_path = Path(args.out)
    if out_path.exists():
        log.info("sentinel %s exists — phase 1.2 already complete, exiting", out_path)
        log.info("contents: %s", out_path.read_text())
        return 0

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # D1 winner cfg — most fields are already the params.yaml default.
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True

    # The single experimental change vs D1.
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"

    # Speed knob — not part of cfg-hash.
    cfg["train"]["num_workers"] = 2

    log.info(
        "Phase 1.2 cfg: encoder=%s clahe=%s boundary=%.2f roi_crop=%s batch=%d lr_dec=%g",
        cfg["train"]["encoder_name"],
        cfg["train"]["preprocess"]["clahe_mode"],
        cfg["train"]["loss"]["boundary_lambda"],
        cfg["train"]["preprocess"]["roi_crop"],
        cfg["train"]["batch_size"],
        cfg["train"]["lr_dec"],
    )

    result = run(cfg, use_cache=not args.no_cache)
    log.info("Phase 1.2 done — best_val_dice=%.3f run_dir=%s", result["best_val_dice"], result["run_dir"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("wrote %s — comparison target D1=0.655", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
