"""RB-UNet (EncoderUNet) inference helpers for the FastAPI deploy.

Wraps :class:`ai.inference.predictor.Predictor` so the deploy can call our
semantic-segmentation model on raw image bytes (no clean_index row).
Preprocessing matches ``ai.training.dataset.preprocess_case`` with
``roi_crop="off"`` so train/inference parity holds.

Default config: 5-fold D2 ensemble (mean softmax + hflip TTA). Per-fold
checkpoint paths come from ``experiments/results/phase1_2_d2_5fold.json``.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training.dataset import IMG_H, IMG_W

VERTEBRA_LABELS = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
assert len(VERTEBRA_LABELS) == NUM_SEG_CLASSES - 1  # 17 fg classes


@dataclass(frozen=True)
class RBUNetConfig:
    """Resolved ensemble config.

    ``run_dirs`` are absolute (or repo-relative) paths to per-fold run dirs,
    each containing ``model.pt`` + ``cfg.json``.
    """
    run_dirs: tuple[Path, ...]
    tta: str = "hflip"


def resolve_d2_5fold(repo_root: Path) -> RBUNetConfig:
    """Load the 5 D2 fold run dirs from the project sentinel."""
    sentinel = repo_root / "experiments" / "results" / "phase1_2_d2_5fold.json"
    if not sentinel.exists():
        raise FileNotFoundError(f"D2 sentinel not found: {sentinel}")
    data = json.loads(sentinel.read_text())
    run_dirs = tuple((repo_root / f["run_dir"]).resolve() for f in data["folds"])
    return RBUNetConfig(run_dirs=run_dirs, tta="hflip")


class RBUNetEnsemble:
    """5-fold (or N-fold) RB-UNet mean-softmax ensemble.

    Loads each fold's Predictor once at startup. Inference runs every fold
    in sequence on the same preprocessed tensor and averages softmax probs.
    """

    def __init__(self, cfg: RBUNetConfig, device: torch.device | None = None) -> None:
        if not cfg.run_dirs:
            raise ValueError("RBUNetConfig.run_dirs is empty")
        self.cfg = cfg
        # Pick device once and force every fold to use it (Predictor would
        # otherwise call get_device() and resolve to DirectML on the dev box).
        self.device = device if device is not None else _pick_device()
        self.predictors: list[Predictor] = [
            Predictor(run_dir, device=self.device) for run_dir in cfg.run_dirs
        ]

    @property
    def n_folds(self) -> int:
        return len(self.predictors)

    @torch.no_grad()
    def predict_probs(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Mean softmax probabilities ``(C, H, W)`` across all folds."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        acc: torch.Tensor | None = None
        for p in self.predictors:
            probs = p.predict_logits(image_tensor, tta=self.cfg.tta)
            acc = probs if acc is None else acc + probs
        return acc / self.n_folds


def _pick_device() -> torch.device:
    """CUDA when available, else CPU. Bypass DirectML on the dev box —
    the deploy target is CUDA EC2 or CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def bytes_to_input_tensor(raw_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    """Decode bytes → (display_uint8_512x512_bgr, input_tensor_1x512x256).

    The display image is the original radiograph resized to 512×512 BGR so
    the response can be drawn on a square canvas matching the YOLO endpoint.
    The input tensor is the model's expected (1, IMG_H, IMG_W) float in
    [0, 1], produced by the same PIL BILINEAR + ``/255`` path as the
    trainer's ``preprocess_case``.
    """
    if not raw_bytes:
        raise ValueError("empty file")
    with Image.open(io.BytesIO(raw_bytes)) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)

    # Display (BGR 512×512 for overlay rendering)
    display_pil = Image.fromarray(gray).resize((512, 512), Image.BILINEAR)
    display_bgr = cv2.cvtColor(np.array(display_pil), cv2.COLOR_GRAY2BGR)

    # Model input (IMG_H × IMG_W = 512 × 256, BILINEAR)
    model_pil = Image.fromarray(gray).resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(model_pil, dtype=np.uint8).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return display_bgr, tensor


def probs_to_instances(probs: torch.Tensor, out_hw: tuple[int, int] = (512, 512)) -> list[dict]:
    """Convert (C, H, W) softmax probs to per-class instance dicts.

    For each foreground class 1..17:
      1. Take argmax mask, isolate pixels of this class.
      2. Run connected components; keep the largest blob (handles small
         speckle without dropping the vertebra).
      3. Compute centroid, area, and mean per-pixel class probability as
         confidence.
      4. Resize the binary mask to ``out_hw`` (NEAREST) so it can be passed
         to the shared overlay renderer.

    Returns a list ordered craniocaudal (top→bottom by centroid_y in
    ``out_hw`` coordinates), each item shaped like a YOLO vertebra dict:
    ``{"label", "confidence", "centroid_x", "centroid_y", "area_px", "mask",
       "source"}``.
    """
    probs_np = probs.detach().cpu().numpy()  # (C, H, W)
    argmax = probs_np.argmax(axis=0).astype(np.uint8)  # (H, W) class ids 0..17
    h_in, w_in = argmax.shape
    out_h, out_w = out_hw

    instances: list[dict] = []
    for class_id in range(1, NUM_SEG_CLASSES):  # 1..17
        class_mask = (argmax == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        n_lab, labels = cv2.connectedComponents(class_mask)
        if n_lab <= 1:
            continue
        # Largest blob by pixel count (skip background label 0)
        sizes = [(int((labels == i).sum()), i) for i in range(1, n_lab)]
        sizes.sort(reverse=True)
        _, best = sizes[0]
        blob = (labels == best).astype(np.uint8)

        prob_layer = probs_np[class_id]  # (H, W)
        confidence = float(prob_layer[blob == 1].mean())

        # Resize mask + recompute centroid in display space
        mask_out = cv2.resize(blob, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask_out)
        if len(ys) == 0:
            continue
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        area_px = int(mask_out.sum())

        instances.append({
            "label": VERTEBRA_LABELS[class_id - 1],
            "confidence": round(confidence, 4),
            "centroid_x": round(centroid_x, 1),
            "centroid_y": round(centroid_y, 1),
            "area_px": area_px,
            "mask": mask_out,
            "source": "rbunet_d2_5fold",
        })

    instances.sort(key=lambda v: v["centroid_y"])
    return instances
