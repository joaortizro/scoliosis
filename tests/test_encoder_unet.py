"""Smoke tests for the EncoderUNet backbone variants.

Each backbone must (a) instantiate from non-pretrained weights, (b)
forward (1, 1, 512, 256) → (1, 18, 512, 256), (c) expose
``encoder_params`` / ``decoder_params`` whose unions equal the full
parameter list.
"""

from __future__ import annotations

import torch

from ai.models.architectures.encoder_unet import EncoderUNet, _BACKBONE_NAMES


def _expect_io(encoder: str) -> None:
    m = EncoderUNet(in_ch=1, num_classes=18, pretrained=False, encoder_name=encoder)
    x = torch.randn(1, 1, 512, 256)
    y = m(x)
    assert y.shape == (1, 18, 512, 256), (encoder, y.shape)


def test_resnet18_io() -> None:
    _expect_io("resnet18")


def test_resnet34_io() -> None:
    _expect_io("resnet34")


def test_txrv_resnet50_io_no_weights() -> None:
    _expect_io("txrv-resnet50")


def test_param_groups_partition() -> None:
    """encoder_params + decoder_params must cover all model params (no leftover)."""
    m = EncoderUNet(in_ch=1, num_classes=18, pretrained=False, encoder_name="resnet34")
    enc = list(m.encoder_params())
    dec = list(m.decoder_params())
    enc_ids = {id(p) for p in enc}
    dec_ids = {id(p) for p in dec}
    all_ids = {id(p) for p in m.parameters()}
    assert enc_ids.isdisjoint(dec_ids)
    assert (enc_ids | dec_ids) == all_ids


def test_backbone_registry_complete() -> None:
    assert "resnet18" in _BACKBONE_NAMES
    assert "resnet34" in _BACKBONE_NAMES
    assert "txrv-resnet50" in _BACKBONE_NAMES
