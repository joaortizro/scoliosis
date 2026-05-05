"""EncoderUNet — single-task segmentation network with a swappable backbone.

Three encoders supported:

- ``resnet18`` / ``resnet34``: torchvision ImageNet-pretrained.
  Stage channel sequence ``[64, 64, 128, 256, 512]``.
- ``txrv-resnet50``: torchxrayvision ResNet-50, ImageNet-style stem
  trained on 5 chest-X-ray datasets (PadChest + NIH + RSNA + SIIM +
  VinDr). Native 1-channel grayscale stem. Stage channels
  ``[64, 256, 512, 1024, 2048]``. Phase 1.1 of the Dice 0.643 → 0.80
  plan; expected gain +1–4 % from domain-matched pretraining.

The decoder mirrors classic UNet: 4 up-convolutions with skip
connections at H/16, H/8, H/4, H/2. Channel counts in the upconv
chain are derived from the encoder's ``stage_channels`` so that adding
new backbones is a one-place change in :func:`_build_encoder`.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ai.models.architectures.base_model import BaseModel


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    """Up-convolution + skip concat + double conv. Channels parameterized."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Halve channels via transposed conv, then concat with skip and reduce.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _conv_block(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x, skip], dim=1))


# Stage channel signatures: [stem_out, layer1, layer2, layer3, layer4].
_TORCHVISION_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
}

_TXRV_BACKBONES = {
    "txrv-resnet50": ("resnet50-res512-all", [64, 256, 512, 1024, 2048]),
}

_BACKBONE_NAMES = list(_TORCHVISION_BACKBONES) + list(_TXRV_BACKBONES)


def _build_encoder(
    encoder_name: str, pretrained: bool, in_ch: int
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, list[int]]:
    """Return ``(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, stage_channels)``.

    Splitting the encoder into eight named pieces keeps :func:`forward`
    readable across backbones. ``stage_channels`` is a 5-tuple of the
    out-channel counts at the points where skip connections are taken.
    """
    if encoder_name in _TORCHVISION_BACKBONES:
        factory, default_weights, channels = _TORCHVISION_BACKBONES[encoder_name]
        weights = default_weights if pretrained else None
        backbone = factory(weights=weights)

        # Replace stem to accept ``in_ch``.
        original = backbone.conv1
        conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_ch == 1:
            conv1.weight.data = original.weight.data.mean(dim=1, keepdim=True)
        elif pretrained and in_ch == 3:
            conv1.weight.data = original.weight.data

        return (
            conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            channels,
        )

    if encoder_name in _TXRV_BACKBONES:
        weight_id, channels = _TXRV_BACKBONES[encoder_name]

        if pretrained:
            # TXRV ships its own ResNet that accepts 1-channel input
            # natively, with weights from PadChest+NIH+RSNA+SIIM+VinDr.
            import torchxrayvision as xrv

            tx = xrv.models.ResNet(weights=weight_id)
            m = tx.model
            if in_ch != 1:
                raise ValueError("txrv-resnet50 with pretrained=True only supports in_ch=1")
        else:
            # Tests/no-weights path — fall back to torchvision's stock
            # ResNet-50 with a 1-channel stem. Channel sequence is
            # identical, so the decoder doesn't care which weights are loaded.
            backbone = models.resnet50(weights=None)
            stem = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone.conv1 = stem
            m = backbone

        return m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4, channels

    raise ValueError(
        f"encoder_name must be one of {_BACKBONE_NAMES}, got {encoder_name!r}"
    )


class EncoderUNet(BaseModel):
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 18,
        pretrained: bool = True,
        dropout: float = 0.0,
        encoder_name: str = "resnet34",
    ):
        super().__init__()

        (
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            stage_channels,
        ) = _build_encoder(encoder_name, pretrained, in_ch)
        self.stage_channels = stage_channels  # [stem, l1, l2, l3, l4]

        # Decoder channel widths: pick a reasonable output for each up step.
        # Standard UNet halves at each step from the bottleneck downward.
        c_stem, c1, c2, c3, c4 = stage_channels
        d3 = max(c3, 64)
        d2 = max(c2, 64)
        d1 = max(c1, 64)
        d0 = 64

        # Each _Up: in_ch (from previous stage), skip_ch (from encoder), out_ch
        self.up1 = _Up(c4, c3, d3)
        self.up2 = _Up(d3, c2, d2)
        self.up3 = _Up(d2, c1, d1)
        self.up4 = _Up(d1, c_stem, d0)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.drop = nn.Dropout2d(dropout)
        self.seg_head = nn.Conv2d(d0, num_classes, 1)

        self._encoder_modules = [
            self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4,
        ]
        self._decoder_modules = [
            self.up1, self.up2, self.up3, self.up4, self.final_up, self.drop, self.seg_head,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.relu(self.bn1(self.conv1(x)))   # (B, c_stem, H/2, W/2)
        x_pool = self.maxpool(x0)                  # (B, c_stem, H/4, W/4)
        s1 = self.layer1(x_pool)                   # (B, c1, H/4, W/4)
        s2 = self.layer2(s1)                       # (B, c2, H/8, W/8)
        s3 = self.layer3(s2)                       # (B, c3, H/16, W/16)
        bridge = self.layer4(s3)                   # (B, c4, H/32, W/32)

        y = self.up1(bridge, s3)
        y = self.up2(y, s2)
        y = self.up3(y, s1)
        y = self.up4(y, x0)
        y = self.final_up(y)
        y = self.drop(y)
        return self.seg_head(y)

    def encoder_params(self) -> Iterator[nn.Parameter]:
        for mod in self._encoder_modules:
            yield from mod.parameters()

    def decoder_params(self) -> Iterator[nn.Parameter]:
        for mod in self._decoder_modules:
            yield from mod.parameters()
