"""Calibration test — corner-based Wu/BoostNet Cobb vs MaIA's `angulo_cobb_deg`.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §3 sanity gate (b).

Path B uses a corner-based Wu/BoostNet pairwise-max Cobb as the primary GT
(literature SOTA standard). v2's `angulo_cobb_deg` is computed via Landinez
et al.'s curve-tangent algorithm — a different valid algorithmic Cobb
definition. This test verifies the two algorithms are CONSISTENT (high
correlation, bounded systematic offset) — not identical, since they measure
different geometric properties.

Test passes when:
    Pearson r ≥ 0.80
    |mean systematic offset| < 10°
    fraction of finite predictions ≥ 95%
    n ≥ 100 trainable scoliosis cases

Calibration data is also useful for the thesis methodology chapter
(quantifies the algorithmic gap for cross-paper comparison).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PIL import Image

from ai.evaluation.cobb_endplate import cobb_from_keypoints_endplate
from ai.preprocessing.keypoints import multiclass_mask_to_keypoints

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"

# Calibration thresholds from spec §3 sanity gate (b)
MIN_CORRELATION = 0.80
MAX_OFFSET_DEG = 10.0
MIN_FINITE_FRACTION = 0.95
MIN_N_CASES = 100


def _load_trainable_scoliosis() -> pd.DataFrame:
    if not CLEAN_INDEX.exists():
        pytest.skip(f"clean_index.csv not present at {CLEAN_INDEX}")
    df = pd.read_csv(CLEAN_INDEX)
    return df[
        (df["category"] == "Scoliosis")
        & (df["target_vertebrae_count"] >= 14)
        & (df["status"].isin(["ok", "warn"]))
        & (df["cobb_angle_deg"].notna())
    ].reset_index(drop=True)


def _compute_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    preds = np.full(len(df), np.nan, dtype=np.float64)
    gts = df["cobb_angle_deg"].to_numpy(dtype=np.float64)
    for i, row in df.iterrows():
        mask_path = Path(row["multiclass_mask_path"])
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path))
        kps = multiclass_mask_to_keypoints(mask)
        cobb, _ = cobb_from_keypoints_endplate(kps)
        preds[i] = cobb
    return preds, gts


def test_calibration_correlation_with_maia() -> None:
    df = _load_trainable_scoliosis()
    assert len(df) >= MIN_N_CASES, f"expected >={MIN_N_CASES} trainable scoliosis cases, got {len(df)}"

    preds, gts = _compute_predictions(df)
    valid = np.isfinite(preds) & np.isfinite(gts)
    finite_fraction = float(valid.sum() / len(df))

    pearson_r = float(np.corrcoef(preds[valid], gts[valid])[0, 1])
    offset_deg = float((preds[valid] - gts[valid]).mean())
    abs_offset = abs(offset_deg)
    mae = float(np.abs(preds[valid] - gts[valid]).mean())
    median_err = float(np.median(np.abs(preds[valid] - gts[valid])))

    print("\n=== Phase 3b.1 calibration vs MaIA ===")
    print(f"n_cases={valid.sum()}  finite_frac={finite_fraction:.3f}")
    print(f"Pearson r={pearson_r:.3f}  systematic offset={offset_deg:+.2f}°")
    print(f"MAE={mae:.2f}°  median={median_err:.2f}°")

    assert finite_fraction >= MIN_FINITE_FRACTION, (
        f"only {finite_fraction:.3f} of cases produced finite predictions, "
        f"need >= {MIN_FINITE_FRACTION}"
    )
    assert pearson_r >= MIN_CORRELATION, (
        f"Pearson correlation {pearson_r:.3f} below required {MIN_CORRELATION}. "
        f"Wu/BoostNet Cobb does not align with MaIA's `angulo_cobb_deg` -- "
        f"expected algorithmic gap, not internal-consistency failure."
    )
    assert abs_offset < MAX_OFFSET_DEG, (
        f"systematic offset {offset_deg:+.2f} deg exceeds {MAX_OFFSET_DEG}. "
        f"Possible sign / unit / orientation bug in cobb_endplate."
    )
