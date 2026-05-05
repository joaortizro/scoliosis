"""Train-time vs inference-time preprocessing must match exactly.

If they ever diverge, val Dice will look fine and prod Dice will tank.
We assert byte-for-byte equality on the same row + clahe_mode.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from ai.training.dataset import SpineDataset, preprocess_case

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"

pytestmark = pytest.mark.skipif(
    not CLEAN_INDEX.exists(),
    reason="dataset not pulled (DVC)",
)


@pytest.fixture(scope="module")
def first_row() -> pd.Series:
    df = pd.read_csv(CLEAN_INDEX)
    return df.iloc[0]


@pytest.mark.parametrize("clahe_mode", ["off", "real"])
def test_preprocess_case_deterministic(first_row: pd.Series, clahe_mode: str) -> None:
    a = preprocess_case(first_row, clahe_mode=clahe_mode)
    b = preprocess_case(first_row, clahe_mode=clahe_mode)
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["seg"], b["seg"])


def test_dataset_matches_preprocess_case(first_row: pd.Series) -> None:
    """SpineDataset (no augment) must yield bit-identical tensors to preprocess_case."""
    ds = SpineDataset(
        pd.DataFrame([first_row]),
        augment=False,
        clahe_mode="real",
    )
    img, seg = ds[0]
    direct = preprocess_case(first_row, clahe_mode="real")
    assert torch.equal(img, direct["image"])
    assert torch.equal(seg, direct["seg"])


def test_clahe_off_vs_real_differs(first_row: pd.Series) -> None:
    """Sanity: the two paths actually produce different images."""
    a = preprocess_case(first_row, clahe_mode="off")["image"]
    b = preprocess_case(first_row, clahe_mode="real")["image"]
    assert not torch.equal(a, b)
