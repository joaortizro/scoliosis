"""Inference wrapper for a saved EncoderUNet checkpoint.

Loads ``model.pt`` + ``cfg.json`` from a run dir, sets up the same
preprocessing the trainer used (parity is enforced — both call
``ai.training.dataset.preprocess_case``), and exposes ``predict`` with
optional horizontal-flip TTA.

No rotation TTA: rotations would change Cobb angle, and Cobb is what
we ultimately report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ai.models.architectures.encoder_unet import EncoderUNet
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training.dataset import IMG_H, IMG_W, preprocess_case
from ai.utils import get_device

TTA = Literal["off", "hflip"]


class Predictor:
    """One-shot predictor for a saved run dir.

    Args:
        run_dir: directory containing ``model.pt`` + ``cfg.json``.
        device: optional override; defaults to :func:`ai.utils.get_device`.
    """

    def __init__(self, run_dir: str | Path, device: torch.device | None = None) -> None:
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "model.pt").exists():
            raise FileNotFoundError(f"{self.run_dir}/model.pt not found")
        if not (self.run_dir / "cfg.json").exists():
            raise FileNotFoundError(f"{self.run_dir}/cfg.json not found")

        self.cfg: dict = json.loads((self.run_dir / "cfg.json").read_text())
        self.device = device if device is not None else get_device().device

        train_cfg = self.cfg.get("train", {})
        pre = train_cfg.get("preprocess", {})
        self.clahe_mode = str(pre.get("clahe_mode", "off"))
        self.roi_crop_mode = str(pre.get("roi_crop", "off"))

        self.model = EncoderUNet(
            in_ch=1,
            num_classes=NUM_SEG_CLASSES,
            pretrained=False,
            dropout=float(train_cfg.get("dropout", 0.0)),
            encoder_name=str(train_cfg.get("encoder_name", "resnet34")),
        ).to(self.device)
        state = torch.load(self.run_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, image: torch.Tensor, tta: TTA = "off") -> torch.Tensor:
        """Run model and return ``(C, H, W)`` softmax-averaged probabilities.

        Args:
            image: ``(1, H, W)`` or ``(B, 1, H, W)`` float tensor in ``[0, 1]``.
            tta: ``"off"`` or ``"hflip"``. When ``"hflip"``, the prediction
                is the mean of the identity and horizontally-flipped views.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[1] != 1:
            raise ValueError(f"image must be (1,H,W) or (B,1,H,W), got {tuple(image.shape)}")

        image = image.to(self.device)
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)

        if tta == "hflip":
            flipped = torch.flip(image, dims=[-1])
            f_logits = self.model(flipped)
            f_probs = F.softmax(f_logits, dim=1)
            f_probs = torch.flip(f_probs, dims=[-1])
            probs = 0.5 * (probs + f_probs)
        elif tta != "off":
            raise ValueError(f"unknown tta {tta!r}")

        return probs[0]  # (C, H, W) — first batch element

    @torch.no_grad()
    def predict_mask(self, image: torch.Tensor, tta: TTA = "off") -> torch.Tensor:
        """Predicted argmax mask ``(H, W)``."""
        return self.predict_logits(image, tta=tta).argmax(dim=0)

    def predict_from_row(
        self, row: pd.Series, tta: TTA = "off"
    ) -> dict[str, torch.Tensor]:
        """Run prediction starting from a clean_index row.

        Uses the same ``preprocess_case`` the trainer used (with the
        ``clahe_mode`` and ``roi_crop_mode`` persisted in the run
        cfg). Returns the input image, GT seg, and predicted mask.
        """
        case = preprocess_case(
            row, clahe_mode=self.clahe_mode, roi_crop_mode=self.roi_crop_mode
        )
        image = case["image"]
        seg = case["seg"]
        pred = self.predict_mask(image, tta=tta).detach().cpu()
        return {"image": image, "seg": seg, "pred": pred}
