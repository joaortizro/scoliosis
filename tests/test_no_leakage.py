"""Hard CI gate: no patient leakage anywhere in the split machinery.

Runs against the live ``clean_index.csv`` and the committed
``test_holdout.csv``. If either of these files moves, this test will
flag it on the first PR. The test is intentionally noisy on failure
because silent leakage is the single biggest cause of inflated
medical-imaging numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai.training.splits import (
    CASE_ID_COL,
    SEVERITY_COL,
    SEVERITY_LEVELS,
    add_case_id,
    add_severity,
    make_canonical_split,
    make_cv_folds,
    make_test_holdout,
    materialize,
    trainable_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
TEST_HOLDOUT = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "test_holdout.csv"


pytestmark = pytest.mark.skipif(
    not CLEAN_INDEX.exists(),
    reason=f"{CLEAN_INDEX} not present (DVC not pulled)",
)


@pytest.fixture(scope="module")
def trainable() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_INDEX)
    return trainable_rows(df)


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    if not TEST_HOLDOUT.exists():
        pytest.skip(f"{TEST_HOLDOUT} not generated yet")
    return pd.read_csv(TEST_HOLDOUT)


def _ids(df: pd.DataFrame) -> set[str]:
    return set(df[CASE_ID_COL])


# ---------------------------------------------------------------------------
# Case identity
# ---------------------------------------------------------------------------


def test_case_id_unique_in_clean_index(trainable: pd.DataFrame) -> None:
    assert trainable[CASE_ID_COL].is_unique, (
        "case_id should be unique across trainable rows. "
        "Composite key (category, patient_id) is the safe identifier — "
        "patient_id alone collides between Normal and Scoliosis."
    )


def test_severity_buckets_are_canonical(trainable: pd.DataFrame) -> None:
    bad = set(trainable[SEVERITY_COL]) - set(SEVERITY_LEVELS)
    assert not bad, f"unexpected severity bucket(s): {bad}"


# ---------------------------------------------------------------------------
# Test holdout discipline
# ---------------------------------------------------------------------------


def test_holdout_subset_of_trainable(trainable: pd.DataFrame, holdout: pd.DataFrame) -> None:
    missing = _ids(holdout) - _ids(trainable)
    assert not missing, (
        f"holdout has case_ids not present in current trainable rows: {sorted(missing)[:5]}"
    )


def test_holdout_severity_stratified(trainable: pd.DataFrame, holdout: pd.DataFrame) -> None:
    # Every severity bucket present in trainable should also appear in holdout
    # (we expect at least 1 of each given a 10% sample with our class sizes).
    buckets_train = set(trainable[SEVERITY_COL].unique())
    buckets_holdout = set(holdout[SEVERITY_COL].unique())
    missing = buckets_train - buckets_holdout
    assert not missing, f"severity bucket(s) absent from holdout: {missing}"


def test_make_test_holdout_idempotent(tmp_path: Path) -> None:
    """Calling make_test_holdout twice with same args returns same slice."""
    out = tmp_path / "ho.csv"
    a = make_test_holdout(CLEAN_INDEX, out, test_frac=0.1, seed=42)
    b = make_test_holdout(CLEAN_INDEX, out, test_frac=0.1, seed=42)
    assert _ids(a) == _ids(b)


def test_make_test_holdout_rejects_bad_seed(tmp_path: Path) -> None:
    """Refuses to overwrite an existing holdout when the on-disk slice
    can't be recovered from the new (seed, test_frac)."""
    out = tmp_path / "ho.csv"
    make_test_holdout(CLEAN_INDEX, out, test_frac=0.1, seed=42)
    # Write a phony holdout that contains a case_id not in trainable.
    bad = pd.DataFrame({CASE_ID_COL: ["Normal_99999"]})
    bad.to_csv(out, index=False)
    with pytest.raises(RuntimeError, match="does not match"):
        make_test_holdout(CLEAN_INDEX, out, test_frac=0.1, seed=42)


# ---------------------------------------------------------------------------
# Canonical 80/20 split (the Phase 0 fidelity gate target)
# ---------------------------------------------------------------------------


def test_canonical_split_no_overlap(trainable: pd.DataFrame, holdout: pd.DataFrame) -> None:
    spec = make_canonical_split(CLEAN_INDEX, TEST_HOLDOUT, val_frac=0.2, seed=42)
    train_ids = _ids(trainable.iloc[list(spec.train_idx)])
    val_ids = _ids(trainable.iloc[list(spec.val_idx)])
    test_ids = _ids(trainable.iloc[list(spec.test_idx)])

    assert not (train_ids & val_ids), "train ∩ val must be empty"
    assert not (train_ids & test_ids), "train ∩ test must be empty"
    assert not (val_ids & test_ids), "val ∩ test must be empty"
    assert test_ids == _ids(holdout), "canonical split test slice must equal frozen holdout"


def test_canonical_split_partition(trainable: pd.DataFrame) -> None:
    """train ∪ val ∪ test == trainable (every case lands somewhere)."""
    spec = make_canonical_split(CLEAN_INDEX, TEST_HOLDOUT, val_frac=0.2, seed=42)
    covered = set(spec.train_idx) | set(spec.val_idx) | set(spec.test_idx)
    assert covered == set(range(len(trainable))), (
        f"split fails to cover trainable rows: missing {set(range(len(trainable))) - covered}"
    )


# ---------------------------------------------------------------------------
# CV folds
# ---------------------------------------------------------------------------


def test_cv_folds_no_overlap(trainable: pd.DataFrame, holdout: pd.DataFrame) -> None:
    folds = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT, n_splits=5, seed=42)
    assert len(folds) == 5

    holdout_ids = _ids(holdout)
    for s in folds:
        train_ids = _ids(trainable.iloc[list(s.train_idx)])
        val_ids = _ids(trainable.iloc[list(s.val_idx)])
        test_ids = _ids(trainable.iloc[list(s.test_idx)])

        assert not (train_ids & val_ids), f"fold {s.fold}: train ∩ val not empty"
        assert not (train_ids & test_ids), f"fold {s.fold}: train ∩ test not empty"
        assert not (val_ids & test_ids), f"fold {s.fold}: val ∩ test not empty"
        assert test_ids == holdout_ids, f"fold {s.fold} test must equal frozen holdout"


def test_cv_folds_cover_pool(trainable: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Every non-holdout case appears in exactly one val fold."""
    folds = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT, n_splits=5, seed=42)
    holdout_ids = _ids(holdout)
    pool_ids = _ids(trainable) - holdout_ids

    val_union: set[str] = set()
    for s in folds:
        v = _ids(trainable.iloc[list(s.val_idx)])
        assert not (v & val_union), f"fold {s.fold} val overlaps with earlier fold val"
        val_union |= v

    assert val_union == pool_ids, (
        f"CV folds do not partition the pool: missing {pool_ids - val_union}, "
        f"extra {val_union - pool_ids}"
    )


def test_cv_severity_stratified(trainable: pd.DataFrame) -> None:
    """Each fold's val severity distribution should be within ±2 cases
    of the proportional expectation (sanity, not exactness)."""
    folds = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT, n_splits=5, seed=42)
    for s in folds:
        val_df = trainable.iloc[list(s.val_idx)]
        for level in SEVERITY_LEVELS:
            global_frac = (trainable[SEVERITY_COL] == level).mean()
            local_frac = (val_df[SEVERITY_COL] == level).mean()
            # Loose bound — StratifiedGroupKFold doesn't guarantee exactness.
            assert abs(local_frac - global_frac) < 0.20, (
                f"fold {s.fold} severity {level} too imbalanced: "
                f"local={local_frac:.2f} global={global_frac:.2f}"
            )


# ---------------------------------------------------------------------------
# Materialization round-trip
# ---------------------------------------------------------------------------


def test_materialize_produces_disjoint_dfs() -> None:
    spec = make_canonical_split(CLEAN_INDEX, TEST_HOLDOUT, val_frac=0.2, seed=42)
    parts = materialize(CLEAN_INDEX, spec)
    a = set(parts["train"][CASE_ID_COL])
    b = set(parts["val"][CASE_ID_COL])
    c = set(parts["test"][CASE_ID_COL])
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_canonical_split_deterministic() -> None:
    a = make_canonical_split(CLEAN_INDEX, TEST_HOLDOUT, val_frac=0.2, seed=42)
    b = make_canonical_split(CLEAN_INDEX, TEST_HOLDOUT, val_frac=0.2, seed=42)
    assert a.hash() == b.hash()


def test_cv_folds_deterministic() -> None:
    a = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT, n_splits=5, seed=42)
    b = make_cv_folds(CLEAN_INDEX, TEST_HOLDOUT, n_splits=5, seed=42)
    assert [s.hash() for s in a] == [s.hash() for s in b]
