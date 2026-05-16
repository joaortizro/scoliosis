"""Partial-FOV evaluation grid.

Walks the (fold × case × coverage_fraction × mode) grid for a trained
checkpoint family and writes per-case + per-bin metrics. Plan:
``2026-05-15_partial_fov_experiment_plan`` §"Coverage fractions" /
§"Metrics".

For each fold, the val cases are loaded via the canonical
``make_cv_folds`` split (the same split the trainer uses). The case's
preprocessed ``(image, seg)`` is fed through
``deterministic_vertical_crop`` for every ``(f, mode)`` combination
before inference; Dice/IoU are computed against the centroid-policy
cropped GT.

Outputs:
- ``--per-case-csv``: one row per (fold, patient_id, f, mode)
- ``--summary-csv``: per-bin mean across cases (and fold-pooled mean)

The Cobb metric is **not** computed here — that's a follow-up once the
Dice gates pass. Plan §"Pass / fail criteria" lists the three Dice
gates that drive go/no-go for the experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.inference.predictor import Predictor
from ai.preprocessing.transforms import deterministic_vertical_crop
from ai.training.dataset import preprocess_case
from ai.training.splits import make_cv_folds, trainable_rows

log = logging.getLogger(__name__)


def _binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    denom = int(p.sum()) + int(g.sum())
    return float(2 * int((p & g).sum()) / denom) if denom else 1.0


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return float(inter / union) if union else 1.0


def _per_class_dice_partial(
    pred: np.ndarray, gt: np.ndarray, n_classes: int = 18
) -> tuple[float, list[int]]:
    """Per-class macro Dice scored only over GT-present classes.

    Returns ``(mean_dice_over_present, present_class_list)``.
    Classes absent from GT do not contribute (NaN → masked out of mean).
    """
    gt_classes = sorted(int(u) for u in np.unique(gt) if u > 0)
    dices: list[float] = []
    for c in gt_classes:
        p = pred == c
        g = gt == c
        denom = int(p.sum()) + int(g.sum())
        dices.append(float(2 * int((p & g).sum()) / denom) if denom else 0.0)
    if not dices:
        return float("nan"), gt_classes
    return float(np.mean(dices)), gt_classes


def _detected_classes(
    pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, n_classes: int = 18
) -> list[int]:
    out: list[int] = []
    for c in range(1, n_classes):
        p = pred == c
        g = gt == c
        denom = int(p.sum()) + int(g.sum())
        if denom == 0:
            continue
        d = float(2 * int((p & g).sum()) / denom)
        if d >= threshold:
            out.append(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--sentinel",
        required=True,
        help="Sentinel JSON from train_partial_fov.py / phase1_2_5fold.py.",
    )
    parser.add_argument(
        "--per-case-csv",
        default="experiments/results/partial_fov_per_case.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="experiments/results/partial_fov_summary.csv",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Per-fold cap for smoke runs; 0 = no cap.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sentinel = json.loads(Path(args.sentinel).read_text())
    with open(args.params) as f:
        cfg = yaml.safe_load(f)

    grid_cfg = cfg["train"]["eval_partial_fov"]
    f_values = [float(v) for v in grid_cfg["coverage_fractions"]]
    modes = [str(m) for m in grid_cfg["modes"]]
    seed = int(grid_cfg.get("random_seed", 42))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("device=%s f_values=%s modes=%s", device, f_values, modes)

    clean_index = Path(cfg["data"]["clean_index"])
    test_holdout = Path(cfg["data"]["test_holdout"])
    splits = make_cv_folds(
        clean_index_csv=str(clean_index),
        test_holdout_csv=str(test_holdout),
        n_splits=int(cfg["data"].get("cv_folds", 5)),
        seed=int(cfg["data"]["random_seed"]),
    )
    full_df = pd.read_csv(clean_index)
    pool = trainable_rows(full_df, min_target_count=14)

    rows: list[dict] = []
    t_start = time.time()
    rng = np.random.default_rng(seed)

    for fold_entry in sentinel["folds"]:
        fold = int(fold_entry["fold"])
        run_dir = REPO / fold_entry["run_dir"]
        log.info("fold %d run_dir=%s", fold, run_dir)
        predictor = Predictor(run_dir, device=device)
        spec = splits[fold]
        val_df = pool.loc[list(spec.val_idx)].reset_index(drop=True)
        if args.limit_cases > 0:
            val_df = val_df.head(args.limit_cases)

        for case_idx, (_, row) in enumerate(val_df.iterrows()):
            case = preprocess_case(
                row,
                clahe_mode=predictor.clahe_mode,
                roi_crop_mode=predictor.roi_crop_mode,
            )
            base_image = case["image"]
            base_seg = case["seg"]
            patient_id = int(row["patient_id"])

            for f in f_values:
                for mode in modes:
                    img_c, seg_c = deterministic_vertical_crop(
                        base_image, base_seg, f=f, mode=mode, rng=rng,
                    )
                    pred = predictor.predict_mask(img_c, tta="off").detach().cpu().numpy().astype(np.int32)
                    gt = seg_c.cpu().numpy().astype(np.int32)
                    d_bin = _binary_dice(pred, gt)
                    iou_bin = _binary_iou(pred, gt)
                    d_mc_partial, present = _per_class_dice_partial(pred, gt)
                    detected = _detected_classes(pred, gt)
                    rows.append({
                        "fold": fold,
                        "patient_id": patient_id,
                        "category": row.get("category"),
                        "f": f,
                        "mode": mode,
                        "binary_dice": d_bin,
                        "binary_iou": iou_bin,
                        "mc_dice_partial": d_mc_partial,
                        "n_gt_present": len(present),
                        "n_detected": len(detected),
                    })

            if (case_idx + 1) % 5 == 0:
                log.info(
                    "  fold %d case %d/%d done (elapsed=%.1fs)",
                    fold, case_idx + 1, len(val_df), time.time() - t_start,
                )

    df = pd.DataFrame(rows)
    per_case_path = Path(args.per_case_csv)
    per_case_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_case_path, index=False)
    log.info("wrote %s (%d rows)", per_case_path, len(df))

    summary = (
        df.groupby(["f", "mode"], as_index=False)
        .agg(
            n=("binary_dice", "count"),
            mean_binary_dice=("binary_dice", "mean"),
            mean_binary_iou=("binary_iou", "mean"),
            mean_mc_dice_partial=("mc_dice_partial", "mean"),
            mean_n_detected=("n_detected", "mean"),
        )
        .round(4)
    )
    summary_path = Path(args.summary_csv)
    summary.to_csv(summary_path, index=False)
    log.info("wrote %s\n%s", summary_path, summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
