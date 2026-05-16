"""Phase 1.2 — ResNet-34 Phase 0 winner + training-time ROI mask crop.

Tests ROI-crop's marginal contribution on top of the Phase 0 winner
(EMA on, ResNet-34 ImageNet pretrain). The single override vs the
Phase 0 winner is ``preprocess.roi_crop = "from_mask"``. Comparison
target: D1 single-split val Dice = 0.655.

**Phase 0 winner cfg is read from `experiments/results/phase0_summary.json`**
(advisor-filed item B from 2026-05-09 — Phase 1.x scripts now consistently
read the summary, matching ``scripts/phase1_1_txrv.py``). Picks the
max-Dice entry's ``clahe_mode`` and ``boundary_lambda``. On the DirectML
phase0 summary the winner is D1 (``clahe=off``, ``boundary_lambda=0.05``,
dice 0.6546); reading the file produces the same cfg as the prior
hardcoded values, so this change is a no-op on cfg-hash (cache stays
valid).

Caveat re. EC2 phase0 rerun (documented in the wiki under
``2026-05-08_phase0_ec2_rerun``): a separate EC2 Phase 0 rerun found D2
(``clahe=real``, ``λ=0.10``) was the EC2-only winner at 0.642 — but that
result is NOT in `phase0_summary.json` (the canonical DirectML summary).
The Phase 1.2 5-fold result (0.6946 ± 0.0205) already exceeds any Phase 0
number by +0.04 to +0.05 due to the ``roi_crop=from_mask`` addition, so
the DirectML-vs-EC2 winner choice has not bottlenecked production.

``num_workers`` is also set, but it is not part of the cfg-hash, so it
does not invalidate any cached run — purely a runtime speed knob.

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
        "--phase0-summary",
        default="experiments/results/phase0_summary.json",
        help="Phase 0 summary JSON; picks max-Dice entry's clahe_mode + boundary_lambda.",
    )
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

    with open(args.phase0_summary) as f:
        ph0 = json.load(f)
    best = max(ph0["results"], key=lambda r: r["best_val_dice"])
    log.info(
        "Phase 0 winner: %s dice=%.4f clahe=%s boundary=%.3f",
        best["name"], best["best_val_dice"], best["clahe_mode"], best["boundary_lambda"],
    )

    # Phase 0 winner cfg (read from summary; see module docstring for the
    # EC2 D2 caveat). On the canonical DirectML summary this resolves to
    # D1 (clahe=off, boundary=0.05) — same as the prior hardcoded values,
    # so cfg-hash and cache are stable.
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = best["clahe_mode"]
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["loss"]["boundary_lambda"] = float(best["boundary_lambda"])
    cfg["train"]["ema"]["enabled"] = True

    # The single experimental change vs D1.
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"

    # Speed knob — not part of cfg-hash.
    cfg["train"]["num_workers"] = 3

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
