from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ai.preprocessing.keypoints import multiclass_mask_to_keypoints
from ai.preprocessing.segmentation import remap_to_target_classes

TARGET_IDS_V2 = tuple(range(1, 18))
IMG_H = 512
IMG_W = 256


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im, dtype=np.uint8)


def resize_image(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def resize_mask(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST), dtype=np.uint8)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0)


def to_seg_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.int64))


def to_kps_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def preprocess_case(
    row: pd.Series,
    target_ids: tuple[int, ...] = TARGET_IDS_V2,
    h: int = IMG_H,
    w: int = IMG_W,
    clahe_mode: str = "off",
    roi_crop_mode: str = "off",
    roi_pad_frac: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Load + resize + remap one row of clean_index.

    ``clahe_mode`` is "off" or "real" (Phase 0.3). When "real",
    applies real CLAHE on the resized grayscale before normalization.

    ``roi_crop_mode`` (Phase 1.2):
      - "off" — full image, default;
      - "from_mask" — crop the union bbox of GT vertebra pixels +
        ``roi_pad_frac`` per side, then resize to (h, w). Used at
        training time on MaIA.
      - "from_yolo" — gated until Roboflow asset is confirmed.

    Deterministic for a given ``(row, clahe_mode, roi_crop_mode)``
    triple so training and inference share the same path
    (parity invariant).
    """
    image_raw = read_gray(Path(row["image_path"]))
    mask_raw = read_mask(Path(row["multiclass_mask_path"]))

    if roi_crop_mode == "off":
        image_np = resize_image(image_raw, h, w)
        mask_np = resize_mask(mask_raw, h, w)
    elif roi_crop_mode == "from_mask":
        from ai.preprocessing.roi_crop import roi_from_mask
        # Crop in raw coords, then resize. The remap_to_target_classes
        # call below can take a sub-region directly because every
        # pixel value still maps the same way.
        top, bottom, left, right = roi_from_mask(mask_raw, pad_frac=roi_pad_frac)
        image_np = resize_image(image_raw[top:bottom, left:right], h, w)
        mask_np = resize_mask(mask_raw[top:bottom, left:right], h, w)
    elif roi_crop_mode == "from_yolo":
        from ai.preprocessing.roi_crop import roi_from_yolo
        roi_from_yolo()  # raises NotImplementedError until gate
        raise AssertionError("unreachable")
    else:
        raise ValueError(f"roi_crop_mode must be 'off' | 'from_mask' | 'from_yolo', got {roi_crop_mode!r}")

    seg_np = remap_to_target_classes(mask_np, target_ids=target_ids)
    kps_np = multiclass_mask_to_keypoints(mask_np, target_ids=target_ids)

    img_tensor = to_image_tensor(normalize_image(image_np))
    if clahe_mode == "real":
        from ai.training.augmentation import clahe_real
        img_tensor = clahe_real(img_tensor)
    elif clahe_mode != "off":
        raise ValueError(f"clahe_mode must be 'off' or 'real', got {clahe_mode!r}")

    return {
        "image": img_tensor,
        "seg": to_seg_tensor(seg_np),
        "kps": to_kps_tensor(kps_np),
    }


class SpineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = False,
        augment_fn: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        target_ids: tuple[int, ...] = TARGET_IDS_V2,
        clahe_mode: str = "off",
        roi_crop_mode: str = "off",
    ):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.augment_fn = augment_fn
        self.target_ids = target_ids
        self.clahe_mode = clahe_mode
        self.roi_crop_mode = roi_crop_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        case = preprocess_case(
            self.df.iloc[i],
            target_ids=self.target_ids,
            clahe_mode=self.clahe_mode,
            roi_crop_mode=self.roi_crop_mode,
        )
        image, seg = case["image"], case["seg"]
        if self.augment and self.augment_fn is not None:
            image, seg = self.augment_fn(image, seg)
        return image, seg
