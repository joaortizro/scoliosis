"""EncoderUNet training loop — moved from notebooks/sandbox/model_primer_v3_corrected.ipynb.

Phase 0 of the Dice 0.643 → 0.80 plan. The notebook version is the
fidelity reference (val_dice = 0.643 on seed 42, 80/20 split, augment_v4,
seg_loss_fn). This module reproduces that with EMA, AdamW + weight
decay, dual-LR cosine schedule with optional encoder-frozen warmup,
optional real CLAHE preprocessing, and the run-dir cache.

The trainer is the only place that knows how to load a config and turn
it into a checkpoint. Splits come from :mod:`ai.training.splits`; never
recomputed here.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.evaluation.seg_metrics import DatasetDiceAccumulator
from ai.models.architectures.encoder_unet import EncoderUNet
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training import augmentation as aug
from ai.training.checkpoint import (
    config_hash,
    find_cached_run,
    new_run_dir,
    save_run,
)
from ai.training.dataset import SpineDataset
from ai.training.ema import make_ema
from ai.training.losses import seg_loss_fn
from ai.training.splits import (
    SplitSpec,
    make_canonical_split,
    make_cv_folds,
    materialize,
)
from ai.utils import get_device, set_seed

log = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path("ai/models/checkpoints/encoder_unet")
DEFAULT_CLEAN_INDEX = "data/processed/audit_v2_corrected/clean_index.csv"
DEFAULT_TEST_HOLDOUT = "data/processed/audit_v2_corrected/test_holdout.csv"


# ---------------------------------------------------------------------------
# Augmentation registry
# ---------------------------------------------------------------------------

_AUGMENT_REGISTRY: dict[str, Callable | None] = {
    "off": None,
    "v2": aug.augment_v2,
    "v3": aug.augment_v3,
    "v4": aug.augment_v4,
}


def _resolve_augment(name: str) -> Callable | None:
    if name not in _AUGMENT_REGISTRY:
        raise ValueError(f"unknown augment {name!r}; choices: {list(_AUGMENT_REGISTRY)}")
    return _AUGMENT_REGISTRY[name]


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def build_model(cfg: dict) -> nn.Module:
    train_cfg = cfg["train"]
    return EncoderUNet(
        in_ch=1,
        num_classes=NUM_SEG_CLASSES,
        pretrained=bool(train_cfg["pretrained"]),
        dropout=float(train_cfg["dropout"]),
        encoder_name=str(train_cfg["encoder_name"]),
    )


_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def _make_optimizer(name: str, params, **kwargs) -> torch.optim.Optimizer:
    if name not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choices: {list(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](params, **kwargs)


def build_optimizer_and_scheduler(
    model: nn.Module, cfg: dict, num_epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Two parameter groups + (Adam|AdamW) + cosine. Encoder may be frozen for warmup."""
    train_cfg = cfg["train"]
    warmup = int(train_cfg.get("freeze_encoder_epochs", 0))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()

    if warmup > 0:
        # Freeze encoder; decoder-only optimizer.
        for p in model.encoder_params():
            p.requires_grad = False
        optimizer = _make_optimizer(
            opt_name,
            list(model.decoder_params()),
            lr=float(train_cfg["lr_dec"]),
            weight_decay=weight_decay,
        )
        scheduler = None  # rebuilt after warmup ends
    else:
        optimizer = _make_optimizer(
            opt_name,
            [
                {"params": list(model.encoder_params()), "lr": float(train_cfg["lr_enc"])},
                {"params": list(model.decoder_params()), "lr": float(train_cfg["lr_dec"])},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    return optimizer, scheduler


def _swap_to_full_optimizer(
    model: nn.Module, cfg: dict, remaining_epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Called when encoder warmup ends — unfreeze and rebuild optimizer."""
    train_cfg = cfg["train"]
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    for p in model.encoder_params():
        p.requires_grad = True
    optimizer = _make_optimizer(
        opt_name,
        [
            {"params": list(model.encoder_params()), "lr": float(train_cfg["lr_enc"])},
            {"params": list(model.decoder_params()), "lr": float(train_cfg["lr_dec"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, remaining_epochs)
    )
    return optimizer, scheduler


def build_dataloaders(
    cfg: dict, spec: SplitSpec
) -> tuple[DataLoader, DataLoader]:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    pre_cfg = train_cfg.get("preprocess", {})

    parts = materialize(data_cfg["clean_index"], spec)
    augment_fn = _resolve_augment(str(train_cfg["augment"]))

    clahe_mode = str(pre_cfg.get("clahe_mode", "off"))
    roi_crop_mode = str(pre_cfg.get("roi_crop", "off"))
    train_ds = SpineDataset(
        parts["train"],
        augment=augment_fn is not None,
        augment_fn=augment_fn,
        clahe_mode=clahe_mode,
        roi_crop_mode=roi_crop_mode,
    )
    val_ds = SpineDataset(
        parts["val"],
        augment=False,
        clahe_mode=clahe_mode,
        roi_crop_mode=roi_crop_mode,
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    boundary_lambda: float = 0.0,
    ema: torch.optim.swa_utils.AveragedModel | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    for images, seg_t in loader:
        images = images.to(device)
        seg_t = seg_t.to(device)
        logits = model(images)
        loss = seg_loss_fn(
            logits,
            seg_t,
            NUM_SEG_CLASSES,
            boundary_lambda=boundary_lambda,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    boundary_lambda: float = 0.0,
) -> dict[str, float]:
    """Pooled per-class Dice + mean loss across the val set."""
    model.eval()
    total_loss = 0.0
    n = 0
    acc = DatasetDiceAccumulator(num_classes=NUM_SEG_CLASSES, device=device)
    for images, seg_t in loader:
        images = images.to(device)
        seg_t = seg_t.to(device)
        logits = model(images)
        total_loss += float(
            seg_loss_fn(logits, seg_t, NUM_SEG_CLASSES, boundary_lambda=boundary_lambda).item()
        )
        n += 1
        acc.update(logits, seg_t)
    return {"loss": total_loss / max(1, n), "dice": acc.compute()}


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------


def run(
    cfg: dict,
    spec: SplitSpec | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Train one model end-to-end. Returns a metrics dict.

    If ``spec`` is None, builds the canonical 80/20 split using the
    seed in ``cfg["data"]["random_seed"]``.
    """
    set_seed(int(cfg["data"]["random_seed"]))
    device = get_device().device

    if spec is None:
        spec = make_canonical_split(
            clean_index_csv=cfg["data"]["clean_index"],
            test_holdout_csv=cfg["data"]["test_holdout"],
            val_frac=float(cfg["data"]["val_frac"]),
            seed=int(cfg["data"]["random_seed"]),
        )

    cache_cfg = _cache_keys(cfg, spec)
    if use_cache:
        cached = find_cached_run(cache_cfg, CHECKPOINT_ROOT)
        if cached is not None and (cached / "metrics.json").exists():
            log.info("reusing cached run %s", cached)
            metrics = json.loads((cached / "metrics.json").read_text())
            return {"run_dir": str(cached), **metrics, "cached": True}

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg, spec)

    train_cfg = cfg["train"]
    num_epochs = int(train_cfg["epochs"])
    warmup = int(train_cfg.get("freeze_encoder_epochs", 0))
    boundary_lambda = float(train_cfg.get("loss", {}).get("boundary_lambda", 0.0))
    early = train_cfg.get("early_stop", {})
    patience = int(early.get("patience", 0)) or num_epochs + 1
    min_delta = float(early.get("min_delta", 0.0))

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, num_epochs - warmup)

    ema = None
    if bool(train_cfg.get("ema", {}).get("enabled", False)):
        ema = make_ema(model, decay=float(train_cfg["ema"].get("decay", 0.999))).to(device)

    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_source = "live"
    no_improve = 0

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if warmup > 0 and epoch == warmup + 1:
            log.info(">>> encoder unfrozen at epoch %d", epoch)
            optimizer, scheduler = _swap_to_full_optimizer(
                model, cfg, remaining_epochs=num_epochs - warmup
            )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            boundary_lambda=boundary_lambda,
            ema=ema,
        )
        live_stats = evaluate(model, val_loader, device, boundary_lambda=boundary_lambda)

        if scheduler is not None and (warmup == 0 or epoch > warmup):
            scheduler.step()

        # EMA: reuse train BN stats by copying them into ema.module.
        ema_dice = float("nan")
        if ema is not None:
            torch.optim.swa_utils.update_bn(train_loader, ema, device=device)
            ema_stats = evaluate(ema, val_loader, device, boundary_lambda=boundary_lambda)
            ema_dice = ema_stats["dice"]

        n_groups = len(optimizer.param_groups)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(live_stats["loss"]),
            "val_dice": float(live_stats["dice"]),
            "ema_dice": float(ema_dice) if ema is not None else float("nan"),
            "lr_enc": float(optimizer.param_groups[0]["lr"]) if n_groups > 1 else 0.0,
            "lr_dec": float(optimizer.param_groups[-1]["lr"]),
            "sec": time.time() - t0,
        }
        history.append(row)

        # Best on val Dice; pick whichever of {live, ema} is highest.
        candidates: list[tuple[float, str, dict[str, torch.Tensor]]] = [
            (live_stats["dice"], "live", _detached_state(model)),
        ]
        if ema is not None:
            candidates.append((ema_dice, "ema", _detached_state(ema.module)))
        cand_dice, cand_src, cand_state = max(candidates, key=lambda t: t[0])

        if cand_dice > best_dice + min_delta:
            best_dice = cand_dice
            best_state = cand_state
            best_source = cand_src
            no_improve = 0
        else:
            no_improve += 1

        log.info(
            "epoch %3d/%d  train=%.3f  val=%.3f  dice=%.3f  ema=%.3f  lr_e=%.1e  lr_d=%.1e  (%.1fs)",
            epoch, num_epochs, row["train_loss"], row["val_loss"], row["val_dice"],
            row["ema_dice"], row["lr_enc"], row["lr_dec"], row["sec"],
        )

        if epoch > warmup and no_improve >= patience:
            log.info(">>> early stop at epoch %d (no improvement for %d epochs)", epoch, patience)
            break

    total_time = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_val_dice": float(best_dice),
        "best_source": best_source,
        "stopped_epoch": int(history[-1]["epoch"]) if history else 0,
        "total_time_sec": float(total_time),
        "split_hash": spec.hash(),
        "split_fold": spec.fold,
    }

    run_dir = new_run_dir(cache_cfg, CHECKPOINT_ROOT)
    save_run(run_dir, _detached_state(model), history, cache_cfg, metrics)
    log.info("saved run to %s — best_val_dice=%.3f (source=%s)", run_dir, best_dice, best_source)

    return {"run_dir": str(run_dir), **metrics, "cached": False}


def _cache_keys(cfg: dict, spec: SplitSpec) -> dict:
    """Stable subset of cfg used as the run-dir cache key."""
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    return {
        "data": {
            "clean_index": data_cfg["clean_index"],
            "test_holdout": data_cfg.get("test_holdout"),
            "random_seed": data_cfg["random_seed"],
            "val_frac": data_cfg.get("val_frac"),
        },
        "train": {
            "encoder_name": train_cfg["encoder_name"],
            "pretrained": train_cfg["pretrained"],
            "dropout": train_cfg["dropout"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "lr_enc": train_cfg["lr_enc"],
            "lr_dec": train_cfg["lr_dec"],
            "weight_decay": train_cfg.get("weight_decay", 1e-4),
            "optimizer": train_cfg.get("optimizer", "adamw"),
            "freeze_encoder_epochs": train_cfg.get("freeze_encoder_epochs", 0),
            "augment": train_cfg["augment"],
            "ema": train_cfg.get("ema", {}),
            "preprocess": train_cfg.get("preprocess", {}),
            "loss": train_cfg.get("loss", {}),
            "early_stop": train_cfg.get("early_stop", {}),
        },
        "split": {"fold": spec.fold, "hash": spec.hash()},
    }


def _detached_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Backwards-compatible entry point used by scripts/train.py
# ---------------------------------------------------------------------------


def train(params: dict) -> dict[str, Any]:
    """Compat shim: the old DVC stage calls ``train(params)``.

    Maps the legacy params dict shape onto :func:`run`.
    """
    return run(params)
