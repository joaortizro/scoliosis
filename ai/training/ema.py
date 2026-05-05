"""EMA wrapper using torch.optim.swa_utils.

Live model is the optimization target; the EMA copy is updated after
every optimizer step. Both are validated each epoch and the trainer
checkpoints whichever has higher val Dice.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel


def make_ema(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    """Return an EMA wrapper. PyTorch ≥2.0 exposes ``get_ema_multi_avg_fn``."""
    try:
        from torch.optim.swa_utils import get_ema_multi_avg_fn
        return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    except ImportError:
        # Fallback: manual EMA via avg_fn.
        def _avg(avg: torch.Tensor, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return avg * decay + x * (1.0 - decay)
        return AveragedModel(model, avg_fn=_avg)


def materialize_ema(ema: AveragedModel) -> nn.Module:
    """Return a plain ``nn.Module`` snapshot from the EMA wrapper.

    Useful for validation and checkpointing — the wrapper carries a
    ``module`` attribute that is the underlying model with averaged
    weights and BN buffers in training mode set by ``update_bn``.
    """
    snapshot = copy.deepcopy(ema.module)
    snapshot.load_state_dict(ema.module.state_dict())
    return snapshot
