"""1-epoch smoke test — guards against import / wiring breakage on every PR.

Skipped when the corrected dataset isn't pulled (CI without DVC).
Runs on whatever device ``ai.utils.get_device()`` returns; budget is
< 60 s on CPU on a small subset of cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai.training.trainer import run

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "clean_index.csv"
TEST_HOLDOUT = REPO_ROOT / "data" / "processed" / "audit_v2_corrected" / "test_holdout.csv"

pytestmark = pytest.mark.skipif(
    not CLEAN_INDEX.exists(),
    reason="dataset not pulled (DVC)",
)


def test_one_epoch_smoke(tmp_path, monkeypatch) -> None:
    """One epoch trains, EMA off, no boundary loss, tiny encoder."""
    # Redirect checkpoint root into tmp_path so the smoke run doesn't
    # pollute the repo's checkpoints dir.
    monkeypatch.setattr(
        "ai.training.trainer.CHECKPOINT_ROOT",
        tmp_path / "ckpts",
    )

    params = {
        "data": {
            "clean_index": str(CLEAN_INDEX),
            "test_holdout": str(TEST_HOLDOUT),
            "val_frac": 0.2,
            "random_seed": 42,
        },
        "train": {
            "encoder_name": "resnet18",
            "pretrained": False,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "lr_enc": 1e-4,
            "lr_dec": 1e-3,
            "weight_decay": 1e-4,
            "freeze_encoder_epochs": 0,
            "augment": "off",
            "early_stop": {"patience": 0, "min_delta": 0.0},
            "ema": {"enabled": False, "decay": 0.999},
            "preprocess": {"clahe_mode": "off"},
            "loss": {"boundary_lambda": 0.0},
        },
    }

    result = run(params, use_cache=False)
    assert "best_val_dice" in result
    assert "split_hash" in result
    # Existence of run_dir + canonical artifacts.
    rd = Path(result["run_dir"])
    assert (rd / "model.pt").exists()
    assert (rd / "cfg.json").exists()
    assert (rd / "history.csv").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "RUN.md").exists()
