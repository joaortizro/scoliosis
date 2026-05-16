"""Cobb-MAE evaluation on the Phase 1.2 5-fold checkpoints.

Loads the 5 fold checkpoints saved under
``ai/models/checkpoints/encoder_unet/<TIMESTAMP>_<cfg_hash>/`` (DVC-tracked
since 2026-05-10), runs inference on each fold's val split using the same
preprocessing the trainer used, computes Cobb angle from the predicted
segmentation via ``cobb_from_segmentation_tangent``, and compares to the
ground-truth ``cobb_angle_deg`` column in ``clean_index.csv`` (which is
itself sourced from ``RadiographMetrics/metricas_cobb_resumen_recalculado.csv``).

Reports per-fold MAE / MdAE / RMSE / count, mean +/- std across folds,
SOSORT severity-stratified MAE (mild < 25, moderate 25-40, severe > 40),
and bootstrap 95 % CI on the cross-fold population (n=1000 resamples
of patients with replacement).

Output: ``experiments/results/phase1_2_5fold_cobb_eval.json``.

This is the missing thesis number: Phase 1 has been segmentation-only
(headline 5-fold mean Dice 0.6946 +/- 0.0205), and the Cobb-axis floor
of these checkpoints has never been measured.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ai.evaluation.cobb import cobb_from_segmentation_tangent
from ai.training.dataset import preprocess_case
from ai.training.splits import make_cv_folds, materialize
from ai.training.trainer import build_model

log = logging.getLogger(__name__)


def _eval_one_fold(
    cfg: dict,
    spec,
    run_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Run inference on one fold's val split, return per-case rows."""
    model = build_model(cfg).to(device)
    state_path = run_dir / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {state_path} -- run dvc pull on its .dvc pointer"
        )
    state = torch.load(state_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    pre = cfg["train"].get("preprocess", {})
    clahe_mode = str(pre.get("clahe_mode", "off"))
    roi_crop_mode = str(pre.get("roi_crop", "off"))

    val_df = materialize(cfg["data"]["clean_index"], spec)["val"]

    rows: list[dict] = []
    with torch.no_grad():
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            patient_id = int(row["patient_id"])
            category = str(row["category"])
            gt_cobb_raw = row["cobb_angle_deg"]
            gt_cobb = (
                float(gt_cobb_raw)
                if gt_cobb_raw == gt_cobb_raw and gt_cobb_raw != ""
                else float("nan")
            )

            case = preprocess_case(
                row,
                clahe_mode=clahe_mode,
                roi_crop_mode=roi_crop_mode,
            )
            image = case["image"].unsqueeze(0).to(device)
            logits = model(image)
            if isinstance(logits, dict):
                logits = logits.get("seg", next(iter(logits.values())))
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            pred_cobb = float(cobb_from_segmentation_tangent(pred_mask))

            rows.append({
                "fold": int(spec.fold),
                "patient_id": patient_id,
                "category": category,
                "gt_cobb_deg": gt_cobb,
                "pred_cobb_deg": pred_cobb,
                "abs_err_deg": (
                    abs(pred_cobb - gt_cobb) if gt_cobb == gt_cobb else float("nan")
                ),
            })
    return rows


def _aggregate_fold(rows: list[dict]) -> dict:
    """Per-fold aggregate. Filters to scoliosis cases with GT cobb."""
    df = pd.DataFrame(rows)
    has_gt = df["gt_cobb_deg"].notna()
    sco = df[has_gt].copy()
    abs_err = sco["abs_err_deg"].to_numpy()
    if len(abs_err) == 0:
        return {"n_scoliosis_with_gt": 0}

    mild = sco[sco["gt_cobb_deg"] < 25.0]["abs_err_deg"].to_numpy()
    moderate = sco[(sco["gt_cobb_deg"] >= 25.0) & (sco["gt_cobb_deg"] < 40.0)][
        "abs_err_deg"
    ].to_numpy()
    severe = sco[sco["gt_cobb_deg"] >= 40.0]["abs_err_deg"].to_numpy()

    return {
        "n_total": int(len(df)),
        "n_normal_no_gt": int((~has_gt).sum()),
        "n_scoliosis_with_gt": int(has_gt.sum()),
        "mae_deg": float(abs_err.mean()),
        "mdae_deg": float(np.median(abs_err)),
        "rmse_deg": float(np.sqrt(np.mean(abs_err**2))),
        "severity": {
            "mild_lt25": {
                "n": int(len(mild)),
                "mae_deg": float(mild.mean()) if len(mild) else None,
            },
            "moderate_25_40": {
                "n": int(len(moderate)),
                "mae_deg": float(moderate.mean()) if len(moderate) else None,
            },
            "severe_gte40": {
                "n": int(len(severe)),
                "mae_deg": float(severe.mean()) if len(severe) else None,
            },
        },
    }


def _bootstrap_ci(values: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {"lo": None, "hi": None, "n_resamples": 0}
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"lo": lo, "hi": hi, "n_resamples": n}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--sentinel",
        default="experiments/results/phase1_2_5fold.json",
        help="Phase 1.2 5-fold sentinel that names the 5 run_dirs to evaluate.",
    )
    parser.add_argument(
        "--out",
        default="experiments/results/phase1_2_5fold_cobb_eval.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sentinel = json.loads(Path(args.sentinel).read_text())
    fold_run_dirs = {int(f["fold"]): Path(f["run_dir"]) for f in sentinel["folds"]}
    log.info("found %d fold run dirs in %s", len(fold_run_dirs), args.sentinel)

    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    # Phase 1.2 D1+ROI cfg overrides (must match scripts/phase1_2_5fold.py
    # so build_model() and preprocess_case() see the same shapes / encoder).
    cfg["train"]["encoder_name"] = "resnet34"
    cfg["train"]["preprocess"]["clahe_mode"] = "off"
    cfg["train"]["preprocess"]["normalization"] = "div255"
    cfg["train"]["preprocess"]["roi_crop"] = "from_mask"
    cfg["train"]["loss"]["boundary_lambda"] = 0.05
    cfg["train"]["ema"]["enabled"] = True

    splits = make_cv_folds(
        clean_index_csv=cfg["data"]["clean_index"],
        test_holdout_csv=cfg["data"]["test_holdout"],
        n_splits=int(cfg["data"].get("cv_folds", 5)),
        seed=int(cfg["data"]["random_seed"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    all_rows: list[dict] = []
    fold_summaries: dict[int, dict] = {}
    for spec in splits:
        if spec.fold not in fold_run_dirs:
            log.warning("no run_dir for fold %d, skipping", spec.fold)
            continue
        run_dir = fold_run_dirs[spec.fold]
        log.info("=== fold %d (run_dir=%s) ===", spec.fold, run_dir)
        rows = _eval_one_fold(cfg, spec, run_dir, device)
        all_rows.extend(rows)
        fold_summaries[spec.fold] = _aggregate_fold(rows)
        s = fold_summaries[spec.fold]
        log.info(
            "fold %d: n=%d (sco=%d) MAE=%.2fdeg MdAE=%.2fdeg RMSE=%.2fdeg",
            spec.fold,
            s.get("n_total", 0),
            s.get("n_scoliosis_with_gt", 0),
            s.get("mae_deg", float("nan")),
            s.get("mdae_deg", float("nan")),
            s.get("rmse_deg", float("nan")),
        )

    # Cross-fold aggregate over scoliosis cases with GT.
    df = pd.DataFrame(all_rows)
    sco = df[df["gt_cobb_deg"].notna()].copy()
    abs_err = sco["abs_err_deg"].to_numpy()
    fold_maes = np.array([
        fold_summaries[k]["mae_deg"]
        for k in sorted(fold_summaries)
        if "mae_deg" in fold_summaries[k]
    ])

    mild = sco[sco["gt_cobb_deg"] < 25.0]["abs_err_deg"].to_numpy()
    moderate = sco[(sco["gt_cobb_deg"] >= 25.0) & (sco["gt_cobb_deg"] < 40.0)][
        "abs_err_deg"
    ].to_numpy()
    severe = sco[sco["gt_cobb_deg"] >= 40.0]["abs_err_deg"].to_numpy()

    summary = {
        "phase": "phase1_2_5fold_cobb_eval",
        "source_sentinel": str(args.sentinel),
        "phase1_2_5fold_dice_mean": sentinel["mean_dice"],
        "phase1_2_5fold_dice_std": sentinel["std_dice"],
        "n_folds": int(len(fold_summaries)),
        "n_total": int(len(df)),
        "n_scoliosis_with_gt": int(len(sco)),
        "cross_fold_mae_mean_deg": float(fold_maes.mean()) if len(fold_maes) else None,
        "cross_fold_mae_std_deg": float(fold_maes.std(ddof=0)) if len(fold_maes) else None,
        "pooled_mae_deg": float(abs_err.mean()) if len(abs_err) else None,
        "pooled_mdae_deg": float(np.median(abs_err)) if len(abs_err) else None,
        "pooled_rmse_deg": float(np.sqrt(np.mean(abs_err**2))) if len(abs_err) else None,
        "pooled_mae_bootstrap_95ci_deg": _bootstrap_ci(abs_err) if len(abs_err) else None,
        "severity_stratified": {
            "mild_lt25": {
                "n": int(len(mild)),
                "mae_deg": float(mild.mean()) if len(mild) else None,
                "ci": _bootstrap_ci(mild) if len(mild) else None,
            },
            "moderate_25_40": {
                "n": int(len(moderate)),
                "mae_deg": float(moderate.mean()) if len(moderate) else None,
                "ci": _bootstrap_ci(moderate) if len(moderate) else None,
            },
            "severe_gte40": {
                "n": int(len(severe)),
                "mae_deg": float(severe.mean()) if len(severe) else None,
                "ci": _bootstrap_ci(severe) if len(severe) else None,
            },
        },
        "per_fold": {str(k): v for k, v in sorted(fold_summaries.items())},
        "literature_anchors": {
            "phase1_x_segmentation_floor_v4_5fold": 8.16,
            "human_inter_rater_sd_lit": "3-5 deg (Cobb_interrater_TraumaMeter_2022)",
            "sota_cmae_lit_2024_2026": "2.4-4.2 deg (Mazurowski 2025, SpineNET 2026, Seg4Reg+)",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "5-fold Cobb MAE = %.2fdeg +/- %.2fdeg (mean +/- std across folds), pooled %.2fdeg, n_sco=%d",
        summary["cross_fold_mae_mean_deg"],
        summary["cross_fold_mae_std_deg"],
        summary["pooled_mae_deg"],
        summary["n_scoliosis_with_gt"],
    )
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
