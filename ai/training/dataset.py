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
) -> dict[str, torch.Tensor]:
    image_np = resize_image(read_gray(Path(row["image_path"])), h, w)
    mask_np = resize_mask(read_mask(Path(row["multiclass_mask_path"])), h, w)
    seg_np = remap_to_target_classes(mask_np, target_ids=target_ids)
    kps_np = multiclass_mask_to_keypoints(mask_np, target_ids=target_ids)
    return {
        "image": to_image_tensor(normalize_image(image_np)),
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
    ):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.augment_fn = augment_fn
        self.target_ids = target_ids

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        case = preprocess_case(self.df.iloc[i], target_ids=self.target_ids)
        image, seg = case["image"], case["seg"]
        if self.augment and self.augment_fn is not None:
            image, seg = self.augment_fn(image, seg)
        return image, seg
