"""Train / val / test split discipline for scoliosis segmentation.

The single source of truth for which case lands in which fold. Every
trainer, evaluator, and CV runner imports from this module — recomputing
folds in user code is a leak waiting to happen.

Splitting rules (see plan: Leakage prevention):

- **Group key**: ``(category, patient_id)``. Patient IDs in v2 are local
  per category (Normal_1 and Scoliosis_1 are different people), so the
  composite key is the only safe identifier. We expose it as a synthetic
  ``case_id`` column.

- **Severity bucket**: ``normal`` (no cobb GT) / ``mild`` (<25°) /
  ``moderate`` (<40°) / ``severe`` (≥40°). Used to stratify both the
  test holdout and CV folds so each split has the same severity mix.

- **Test holdout** is created **once** (``make_test_holdout``) and
  written to disk — idempotent: if the file already exists with a
  matching seed/size, it is loaded back unchanged. The trainer never
  reads it; only ``scripts/eval_test.py`` does, and only after all gates
  pass.

- **CV folds** are computed by ``StratifiedGroupKFold`` (sklearn ≥1.3),
  guaranteeing zero overlap between train and val for any fold AND
  approximate severity stratification. Test holdout is excluded by
  construction.

The module emits a deterministic identity hash for every split so the
trainer can assert that the on-disk split matches the in-memory plan
before training starts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CASE_ID_COL = "case_id"
SEVERITY_COL = "severity_bucket"
# Three buckets, not four: the corrected dataset has only 1 case under 25°,
# so a literal mild/moderate/severe split breaks stratification. Collapsing
# mild+moderate into ``low`` keeps every fold populated. ``normal`` =
# no GT cobb (Normal category cases).
SEVERITY_LEVELS = ("normal", "low", "severe")
TRAINABLE_STATUSES = ("ok", "warn")


@dataclass(frozen=True)
class SplitSpec:
    """One (train, val) fold plus the global test holdout.

    Indices are positions in the *trainable* DataFrame returned by
    :func:`trainable_rows`, never raw row numbers from ``clean_index.csv``.
    The fold identifier ``fold == -1`` is reserved for the canonical
    80/20 split used as the Phase 0 fidelity gate.
    """

    fold: int
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    seed: int

    def hash(self) -> str:
        payload = {
            "fold": self.fold,
            "train": list(self.train_idx),
            "val": list(self.val_idx),
            "test": list(self.test_idx),
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Trainable filter + derived columns
# ---------------------------------------------------------------------------


def add_case_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic case_id = ``f\"{category}_{patient_id}\"``.

    Idempotent.
    """
    out = df.copy()
    out[CASE_ID_COL] = out["category"].astype(str) + "_" + out["patient_id"].astype(str)
    return out


def severity_bucket(cobb_deg: float | None) -> str:
    """Cobb severity bucket — three levels (see SEVERITY_LEVELS).

    Cobb < 40° collapses to ``low`` because the corrected dataset has
    only one case under 25°; further sub-bucketing would break
    stratification.
    """
    if cobb_deg is None or pd.isna(cobb_deg):
        return "normal"
    if cobb_deg < 40.0:
        return "low"
    return "severe"


def add_severity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[SEVERITY_COL] = out["cobb_angle_deg"].apply(severity_bucket)
    return out


def trainable_rows(df: pd.DataFrame, min_target_count: int = 14) -> pd.DataFrame:
    """Filter clean_index to trainable cases.

    Keeps rows whose ``status`` is ``ok`` or ``warn`` AND whose
    ``target_vertebrae_count`` ≥ ``min_target_count``. Returns a
    DataFrame with ``case_id`` and ``severity_bucket`` populated and
    rows in deterministic order (sorted by case_id) so the index
    positions are reproducible across runs.
    """
    if "status" not in df.columns or "target_vertebrae_count" not in df.columns:
        raise ValueError("clean_index missing required columns")
    keep = df["status"].isin(TRAINABLE_STATUSES) & (df["target_vertebrae_count"] >= min_target_count)
    out = df.loc[keep].copy()
    out = add_case_id(out)
    out = add_severity(out)
    out = out.sort_values(CASE_ID_COL, kind="stable").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Test holdout (frozen, idempotent)
# ---------------------------------------------------------------------------


def make_test_holdout(
    clean_index_csv: str | Path,
    out_csv: str | Path,
    test_frac: float = 0.10,
    seed: int = 42,
    min_target_count: int = 14,
) -> pd.DataFrame:
    """Create the frozen test holdout once.

    If ``out_csv`` already exists and matches the (seed, test_frac,
    composition hash) declared in its header, it is returned unchanged
    — the gate is intentionally strict so a casual re-run cannot
    silently rotate the test set under us.

    Returns the test slice as a DataFrame (subset of trainable rows
    with the same columns + ``case_id``, ``severity_bucket``).
    """
    out_csv = Path(out_csv)
    df = pd.read_csv(clean_index_csv)
    full = trainable_rows(df, min_target_count=min_target_count)

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        if not _holdout_matches(existing, full, seed, test_frac):
            raise RuntimeError(
                f"{out_csv} exists but does not match (seed={seed}, "
                f"test_frac={test_frac}). Refusing to overwrite a frozen test slice. "
                "If you genuinely intend to rotate the holdout, delete the file by hand."
            )
        return existing

    # Stratified random sample on severity bucket. Group-aware NOT needed
    # here because case_id is already unique per row (one row per case).
    train_val_idx, test_idx = train_test_split(
        np.arange(len(full)),
        test_size=test_frac,
        random_state=seed,
        stratify=full[SEVERITY_COL].to_numpy(),
    )
    holdout = full.iloc[sorted(test_idx)].reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(out_csv, index=False)
    return holdout


def _holdout_matches(
    existing: pd.DataFrame,
    full: pd.DataFrame,
    seed: int,
    test_frac: float,
) -> bool:
    """Reject if the on-disk holdout doesn't reproduce from (seed, test_frac).

    We don't actually re-derive the split (would require running
    train_test_split with the same RNG state); instead we sanity-check
    the size and that every case_id in the file exists in the current
    trainable set. If clean_index.csv was rebuilt and a holdout case
    disappears, that is a real problem and we want loud failure.
    """
    expected_n = max(1, round(test_frac * len(full)))
    if abs(len(existing) - expected_n) > 1:
        return False
    if CASE_ID_COL not in existing.columns:
        return False
    if not set(existing[CASE_ID_COL]).issubset(set(full[CASE_ID_COL])):
        return False
    return True


# ---------------------------------------------------------------------------
# CV folds (stratified group)
# ---------------------------------------------------------------------------


def make_cv_folds(
    clean_index_csv: str | Path,
    test_holdout_csv: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    min_target_count: int = 14,
) -> list[SplitSpec]:
    """Build the K-fold splits over (trainable \\ test_holdout).

    Stratified by ``severity_bucket``; grouped by ``case_id`` so the
    same case never lands in both train and val. Indices are positions
    in the full trainable DataFrame (with the test slice still
    present); the test slice is excluded by construction from both
    ``train_idx`` and ``val_idx``.

    Test indices are the same for every fold (the frozen holdout).
    """
    df = pd.read_csv(clean_index_csv)
    full = trainable_rows(df, min_target_count=min_target_count)

    holdout = pd.read_csv(test_holdout_csv)
    holdout_ids = set(holdout[CASE_ID_COL])
    test_idx = full.index[full[CASE_ID_COL].isin(holdout_ids)].to_numpy()
    train_pool_mask = ~full[CASE_ID_COL].isin(holdout_ids)
    pool = full.loc[train_pool_mask]

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pool_pos = full.index[train_pool_mask].to_numpy()
    splits: list[SplitSpec] = []
    for fold, (tr_local, va_local) in enumerate(
        skf.split(
            X=np.zeros(len(pool)),
            y=pool[SEVERITY_COL].to_numpy(),
            groups=pool[CASE_ID_COL].to_numpy(),
        )
    ):
        train_idx = pool_pos[tr_local]
        val_idx = pool_pos[va_local]
        splits.append(
            SplitSpec(
                fold=fold,
                train_idx=tuple(int(i) for i in train_idx),
                val_idx=tuple(int(i) for i in val_idx),
                test_idx=tuple(int(i) for i in test_idx),
                seed=seed,
            )
        )
    return splits


def make_canonical_split(
    clean_index_csv: str | Path,
    test_holdout_csv: str | Path,
    val_frac: float = 0.2,
    seed: int = 42,
    min_target_count: int = 14,
) -> SplitSpec:
    """80/20 train/val split (excluding test holdout) for the Phase 0 fidelity gate.

    Stratified by severity. Uses ``fold == -1`` as the marker.
    """
    df = pd.read_csv(clean_index_csv)
    full = trainable_rows(df, min_target_count=min_target_count)

    holdout = pd.read_csv(test_holdout_csv)
    holdout_ids = set(holdout[CASE_ID_COL])
    test_idx = full.index[full[CASE_ID_COL].isin(holdout_ids)].to_numpy()

    pool_mask = ~full[CASE_ID_COL].isin(holdout_ids)
    pool_pos = full.index[pool_mask].to_numpy()
    pool = full.loc[pool_mask]

    train_local, val_local = train_test_split(
        np.arange(len(pool)),
        test_size=val_frac,
        random_state=seed,
        stratify=pool[SEVERITY_COL].to_numpy(),
    )
    return SplitSpec(
        fold=-1,
        train_idx=tuple(int(i) for i in pool_pos[train_local]),
        val_idx=tuple(int(i) for i in pool_pos[val_local]),
        test_idx=tuple(int(i) for i in test_idx),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Materialization helpers (used by trainer)
# ---------------------------------------------------------------------------


def materialize(
    clean_index_csv: str | Path,
    spec: SplitSpec,
    min_target_count: int = 14,
) -> dict[str, pd.DataFrame]:
    """Load clean_index, apply trainable filter, slice into train/val/test DataFrames.

    The trainer calls this once at the start of training. The returned
    DataFrames are dropped into ``SpineDataset`` directly.
    """
    df = pd.read_csv(clean_index_csv)
    full = trainable_rows(df, min_target_count=min_target_count)

    return {
        "train": full.iloc[list(spec.train_idx)].reset_index(drop=True),
        "val": full.iloc[list(spec.val_idx)].reset_index(drop=True),
        "test": full.iloc[list(spec.test_idx)].reset_index(drop=True),
    }
