from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import functional as TF

from ai.training.dataset import IMG_H, IMG_W


def _uniform(lo: float, hi: float) -> float:
    return float(torch.empty(1).uniform_(lo, hi).item())


# ---------------------------------------------------------------------------
# v2 transforms (preserved from model_primer_v2)
# ---------------------------------------------------------------------------


def affine_transform(
    image: torch.Tensor,
    seg: torch.Tensor,
    angle_range: tuple[float, float] = (-5.0, 5.0),
    translate_frac: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = _uniform(*angle_range)
    tx = _uniform(-translate_frac, translate_frac) * IMG_W
    ty = _uniform(-translate_frac, translate_frac) * IMG_H
    scale = _uniform(*scale_range)
    image = TF.affine(
        image,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[0.0],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    seg_3d = seg.unsqueeze(0).float()
    seg_3d = TF.affine(
        seg_3d,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[0.0],
        interpolation=TF.InterpolationMode.NEAREST,
    )
    seg = seg_3d.squeeze(0).long()
    return image, seg


def intensity_jitter(
    image: torch.Tensor,
    gain_range: tuple[float, float] = (0.8, 1.2),
    bias_range: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    gain = _uniform(*gain_range)
    bias = _uniform(*bias_range)
    return (image * gain + bias).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# v3 new transforms — all operate on CPU tensors
# ---------------------------------------------------------------------------


def horizontal_flip(
    image: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.flip(image, dims=[-1]), torch.flip(seg, dims=[-1])


def elastic_deform(
    image: torch.Tensor,
    seg: torch.Tensor,
    alpha: float = 30.0,
    sigma: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[-2], image.shape[-1]
    dx = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]

    img_np = image.squeeze(0).numpy()
    seg_np = seg.numpy().astype(np.float64)
    img_warped = map_coordinates(img_np, coords, order=1, mode="reflect").astype(
        np.float32
    )
    seg_warped = map_coordinates(seg_np, coords, order=0, mode="reflect").astype(
        np.int64
    )
    return (
        torch.from_numpy(img_warped).unsqueeze(0),
        torch.from_numpy(seg_warped),
    )


def gaussian_noise(
    image: torch.Tensor,
    std_range: tuple[float, float] = (0.0, 0.05),
) -> torch.Tensor:
    std = _uniform(*std_range)
    return (image + torch.randn_like(image) * std).clamp(0.0, 1.0)


def coarse_dropout(
    image: torch.Tensor,
    n_patches: int = 8,
    patch_size: int = 32,
) -> torch.Tensor:
    img = image.clone()
    _, h, w = img.shape
    for _ in range(n_patches):
        y = int(torch.randint(0, max(1, h - patch_size), (1,)).item())
        x = int(torch.randint(0, max(1, w - patch_size), (1,)).item())
        img[:, y : y + patch_size, x : x + patch_size] = 0.0
    return img


# ---------------------------------------------------------------------------
# v2 composed pipeline (backward compat)
# ---------------------------------------------------------------------------


def augment_v2(
    image: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        image, seg = affine_transform(image, seg)
    if torch.rand(1).item() < 0.5:
        image = intensity_jitter(image)
    return image, seg


# ---------------------------------------------------------------------------
# v3 composed pipeline
# ---------------------------------------------------------------------------


def augment_v3(
    image: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        image, seg = affine_transform(image, seg)
    if torch.rand(1).item() < 0.5:
        image = intensity_jitter(image)
    if torch.rand(1).item() < 0.3:
        image, seg = horizontal_flip(image, seg)
    if torch.rand(1).item() < 0.3:
        image, seg = elastic_deform(image, seg)
    if torch.rand(1).item() < 0.3:
        image = gaussian_noise(image)
    if torch.rand(1).item() < 0.2:
        image = coarse_dropout(image)
    return image, seg


# ---------------------------------------------------------------------------
# v4 new transforms
# ---------------------------------------------------------------------------


def gamma_correction(
    image: torch.Tensor,
    gamma_range: tuple[float, float] = (0.7, 1.5),
) -> torch.Tensor:
    gamma = _uniform(*gamma_range)
    return image.clamp(1e-8, 1.0).pow(gamma)


def gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 5,
    sigma_range: tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    sigma = _uniform(*sigma_range)
    return TF.gaussian_blur(image.unsqueeze(0), kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]).squeeze(0)


def clahe_like(
    image: torch.Tensor,
    clip_range: tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Simplified local contrast enhancement on CPU tensors."""
    img_np = image.squeeze(0).numpy()
    blurred = gaussian_filter(img_np, sigma=8.0)
    local_contrast = img_np - blurred
    gain = _uniform(*clip_range)
    enhanced = blurred + local_contrast * gain
    return torch.from_numpy(np.clip(enhanced, 0.0, 1.0).astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# v4 composed pipeline — stronger augmentation for 122-image dataset
# ---------------------------------------------------------------------------


def augment_v4(
    image: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.7:
        image, seg = affine_transform(
            image, seg,
            angle_range=(-15.0, 15.0),
            translate_frac=0.10,
            scale_range=(0.85, 1.15),
        )
    if torch.rand(1).item() < 0.5:
        image, seg = horizontal_flip(image, seg)
    if torch.rand(1).item() < 0.4:
        image, seg = elastic_deform(image, seg, alpha=40.0, sigma=6.0)
    if torch.rand(1).item() < 0.5:
        image = intensity_jitter(image, gain_range=(0.7, 1.3), bias_range=(-0.15, 0.15))
    if torch.rand(1).item() < 0.3:
        image = gamma_correction(image)
    if torch.rand(1).item() < 0.3:
        image = gaussian_blur(image)
    if torch.rand(1).item() < 0.2:
        image = clahe_like(image)
    if torch.rand(1).item() < 0.3:
        image = gaussian_noise(image, std_range=(0.0, 0.08))
    if torch.rand(1).item() < 0.25:
        image = coarse_dropout(image, n_patches=12, patch_size=40)
    return image, seg
