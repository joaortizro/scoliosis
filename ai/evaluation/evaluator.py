"""DVC ``evaluate`` stage — Dice + post-hoc Cobb MAE on the val split.

Loads the most recent matching checkpoint under
``ai/models/checkpoints/encoder_unet/`` and runs inference on the val
split of the canonical 80/20 split (NOT the test holdout — the test
slice is touched only by ``scripts/eval_test.py`` after all gates
pass). Writes ``experiments/results/metrics.json`` for DVC.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai.evaluation.cobb import cobb_from_segmentation_tangent
from ai.evaluation.seg_metrics import DatasetDiceAccumulator
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training.checkpoint import config_hash, find_cached_run
from ai.training.splits import make_canonical_split, materialize
from ai.training.trainer import CHECKPOINT_ROOT, _cache_keys

log = logging.getLogger(__name__)


def _latest_run_dir() -> Path:
    runs = [p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()] if CHECKPOINT_ROOT.exists() else []
    if not runs:
        raise FileNotFoundError(f"no run dirs under {CHECKPOINT_ROOT}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(params: dict) -> Path:
    """Use the cached run that matches current params if present, else latest."""
    spec = make_canonical_split(
        clean_index_csv=params["data"]["clean_index"],
        test_holdout_csv=params["data"]["test_holdout"],
        val_frac=float(params["data"]["val_frac"]),
        seed=int(params["data"]["random_seed"]),
    )
    cache_cfg = _cache_keys(params, spec)
    cached = find_cached_run(cache_cfg, CHECKPOINT_ROOT)
    return cached if cached is not None else _latest_run_dir()


def evaluate(params: dict) -> dict:
    """Compute val Dice + Cobb MAE for the run matching ``params``.

    Writes ``experiments/results/metrics.json`` (path overridable via
    ``params["evaluate"]["metrics_path"]``).
    """
    run_dir = _resolve_run_dir(params)
    log.info("evaluating %s", run_dir)
    predictor = Predictor(run_dir)
    device = predictor.device

    spec = make_canonical_split(
        clean_index_csv=params["data"]["clean_index"],
        test_holdout_csv=params["data"]["test_holdout"],
        val_frac=float(params["data"]["val_frac"]),
        seed=int(params["data"]["random_seed"]),
    )
    parts = materialize(params["data"]["clean_index"], spec)
    val_df = parts["val"]

    acc = DatasetDiceAccumulator(num_classes=NUM_SEG_CLASSES, device=device)
    cobb_abs_errs: list[float] = []

    for _, row in val_df.iterrows():
        out = predictor.predict_from_row(row, tta="hflip")
        # accumulate Dice using the model's logits — re-run for raw logits
        # (cheaper to just convert mask back to logits-shape one-hot for
        # Dice; pooled metric only needs argmax+gt).
        with torch.no_grad():
            logits_pred = torch.zeros(
                (1, NUM_SEG_CLASSES, *out["pred"].shape), device=device
            )
            logits_pred.scatter_(1, out["pred"].to(device).long().unsqueeze(0).unsqueeze(0), 1.0)
            target = out["seg"].to(device).long().unsqueeze(0)
            acc.update(logits_pred, target)

        # Cobb (only meaningful when GT cobb is non-null — Scoliosis cases).
        gt_cobb = row.get("cobb_angle_deg")
        if gt_cobb is not None and not pd.isna(gt_cobb):
            pred_cobb = cobb_from_segmentation_tangent(out["pred"].numpy())
            cobb_abs_errs.append(abs(float(pred_cobb) - float(gt_cobb)))

    val_dice = acc.compute()
    cobb_mae = float(np.mean(cobb_abs_errs)) if cobb_abs_errs else float("nan")

    metrics = {
        "run_dir": str(run_dir),
        "val_dice": float(val_dice),
        "cobb_mae_deg": cobb_mae,
        "n_val_cases": int(len(val_df)),
        "n_cobb_cases": int(len(cobb_abs_errs)),
    }

    out_path = Path(params.get("evaluate", {}).get("metrics_path", "experiments/results/metrics.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out_path)
    return metrics
