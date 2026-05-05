"""Phase 0 ablation runner — drives the Run B / C / D sweep sequentially.

Runs are sequential because they share the same DirectML / CUDA
device. Logs to ``logs/phase0_<run_name>.log``; final summary written
to ``experiments/results/phase0_summary.json``.

Sweep layout:
- B   : Phase 0 stack (EMA on, CLAHE off, boundary 0)
- C   : Phase 0 + real CLAHE
- D1  : Phase 0 + boundary λ=0.05    (on top of CLAHE winner)
- D2  : Phase 0 + boundary λ=0.10
- D3  : Phase 0 + boundary λ=0.20

Each run respects the cfg-hash cache, so re-running is idempotent.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import yaml

from ai.training.trainer import run

log = logging.getLogger(__name__)

CONFIGS = [
    {
        "name": "B_phase0_stack",
        "ema_enabled": True,
        "clahe_mode": "off",
        "boundary_lambda": 0.0,
    },
    {
        "name": "C_phase0_clahe",
        "ema_enabled": True,
        "clahe_mode": "real",
        "boundary_lambda": 0.0,
    },
    {
        "name": "D1_boundary_005",
        "ema_enabled": True,
        "clahe_mode": "winner",
        "boundary_lambda": 0.05,
    },
    {
        "name": "D2_boundary_010",
        "ema_enabled": True,
        "clahe_mode": "winner",
        "boundary_lambda": 0.10,
    },
    {
        "name": "D3_boundary_020",
        "ema_enabled": True,
        "clahe_mode": "winner",
        "boundary_lambda": 0.20,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", default="experiments/results/phase0_summary.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.params) as f:
        base_cfg = yaml.safe_load(f)
    if args.epochs is not None:
        base_cfg["train"]["epochs"] = args.epochs

    results: list[dict] = []
    clahe_winner = "off"  # set after Run C completes

    t_start = time.time()
    for entry in CONFIGS:
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["ema"]["enabled"] = entry["ema_enabled"]

        clahe = entry["clahe_mode"]
        if clahe == "winner":
            clahe = clahe_winner
        cfg["train"]["preprocess"]["clahe_mode"] = clahe
        cfg["train"]["loss"]["boundary_lambda"] = entry["boundary_lambda"]

        log.info("=== %s (clahe=%s, boundary=%.2f) ===",
                 entry["name"], clahe, entry["boundary_lambda"])
        result = run(cfg, use_cache=not args.no_cache)
        log.info("%s done — best_val_dice=%.3f", entry["name"], result["best_val_dice"])
        results.append({
            "name": entry["name"],
            "clahe_mode": clahe,
            "boundary_lambda": entry["boundary_lambda"],
            "best_val_dice": result["best_val_dice"],
            "best_source": result["best_source"],
            "run_dir": result["run_dir"],
        })

        # After Run C lands, decide CLAHE winner for the boundary sweep.
        if entry["name"] == "C_phase0_clahe":
            run_b = next(r for r in results if r["name"] == "B_phase0_stack")
            run_c = next(r for r in results if r["name"] == "C_phase0_clahe")
            clahe_winner = "real" if run_c["best_val_dice"] > run_b["best_val_dice"] else "off"
            log.info("CLAHE winner: %s (B=%.3f, C=%.3f)",
                     clahe_winner, run_b["best_val_dice"], run_c["best_val_dice"])

    summary = {
        "results": results,
        "total_time_sec": float(time.time() - t_start),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_path)
    for r in results:
        log.info("  %s : %.3f", r["name"], r["best_val_dice"])


if __name__ == "__main__":
    main()
