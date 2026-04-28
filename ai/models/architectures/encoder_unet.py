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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _conv_block(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x, skip], dim=1))


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
}


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

        if encoder_name not in _BACKBONES:
            raise ValueError(f"encoder_name must be one of {list(_BACKBONES)}, got {encoder_name!r}")

        factory, default_weights = _BACKBONES[encoder_name]
        weights = default_weights if pretrained else None
        backbone = factory(weights=weights)

        # Grayscale stem: average pretrained RGB conv1 weights → 1 channel
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained and in_ch == 1:
            self.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )
        elif pretrained and in_ch == 3:
            self.conv1.weight.data = original_conv1.weight.data

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64 ch
        self.layer2 = backbone.layer2  # 128 ch
        self.layer3 = backbone.layer3  # 256 ch
        self.layer4 = backbone.layer4  # 512 ch

        self.up1 = _Up(512, 256, 256)
        self.up2 = _Up(256, 128, 128)
        self.up3 = _Up(128, 64, 64)
        self.up4 = _Up(64, 64, 64)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.drop = nn.Dropout2d(dropout)
        self.seg_head = nn.Conv2d(64, num_classes, 1)

        self._encoder_modules = [
            self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4,
        ]
        self._decoder_modules = [
            self.up1, self.up2, self.up3, self.up4, self.final_up, self.drop, self.seg_head,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)                 # (B, 64, H/4, W/4)
        s1 = self.layer1(x_pool)                  # (B, 64, H/4, W/4)
        s2 = self.layer2(s1)                      # (B, 128, H/8, W/8)
        s3 = self.layer3(s2)                       # (B, 256, H/16, W/16)
        bridge = self.layer4(s3)                   # (B, 512, H/32, W/32)

        # Decoder
        y = self.up1(bridge, s3)                   # (B, 256, H/16, W/16)
        y = self.up2(y, s2)                        # (B, 128, H/8, W/8)
        y = self.up3(y, s1)                        # (B, 64, H/4, W/4)
        y = self.up4(y, x0)                        # (B, 64, H/2, W/2)
        y = self.final_up(y)                       # (B, 64, H, W)
        y = self.drop(y)
        return self.seg_head(y)

    def encoder_params(self) -> Iterator[nn.Parameter]:
        for mod in self._encoder_modules:
            yield from mod.parameters()

    def decoder_params(self) -> Iterator[nn.Parameter]:
        for mod in self._decoder_modules:
            yield from mod.parameters()
