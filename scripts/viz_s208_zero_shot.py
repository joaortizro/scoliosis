"""Render Scoliosis_208 — image, GT, predictions from fold0/D1/D2 — to a single PNG.

Saves to /tmp/scoliosis_viz/s208_zero_shot.png (gitignored path, per project rule:
patient-data images must not be committed). The user can inspect locally to settle
whether the zero-shot collapse on S_208 is annotation-mismatch or real failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from ai.evaluation.seg_metrics import confusion_per_class
from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES


def vertebra_cmap() -> ListedColormap:
    base = plt.colormaps["tab20"].colors + plt.colormaps["tab20b"].colors
    colors = [(0, 0, 0, 0)] + list(base[:17])
    return ListedColormap(colors)


def per_case_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """Binary + multi-class macro Dice, and class-present count, for one case."""
    pred_b = pred.long().unsqueeze(0)
    target_b = target.long().unsqueeze(0)
    c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
    tp, fp, fn = c["tp"].double(), c["fp"].double(), c["fn"].double()
    case_dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    present = (tp + fn) > 0
    fg_present = present[1:]
    macro = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")

    pred_fg = (pred > 0)
    gt_fg = (target > 0)
    inter = (pred_fg & gt_fg).sum().double()
    union = pred_fg.sum().double() + gt_fg.sum().double()
    binary = float((2 * inter / (union + 1e-9)).item())
    n_present = int(fg_present.sum().item())
    return binary, macro, n_present


def main() -> None:
    models = [
        ("fold0 (v2-only, ep 83, patience=20) — zero-shot", "ai/models/checkpoints/encoder_unet/20260509_194823_b41714d16d325371"),
        ("D1 (v2_corrected_x2, ep 43, patience=10) — zero-shot", "ai/models/checkpoints/encoder_unet/20260517_035932_34b773bbd50ff3b6"),
        ("D2 (saw S_208 in training, ep 71) — MEMORIZATION", "ai/models/checkpoints/encoder_unet/20260517_050245_1b8fe848ffa7b4fb"),
    ]

    rows = pd.read_csv("data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv")
    s208 = rows[(rows["category"] == "Scoliosis") & (rows["patient_id"] == 208)].iloc[0]

    cmap = vertebra_cmap()
    fig, axes = plt.subplots(len(models), 4, figsize=(18, 4.5 * len(models)))

    for row_idx, (label, run_dir) in enumerate(models):
        predictor = Predictor(run_dir, device=torch.device("cpu"))
        out = predictor.predict_from_row(s208, tta="off")
        img = out["image"].squeeze().cpu().numpy()
        gt = out["seg"].cpu().numpy().astype(np.int32)
        pred = out["pred"].cpu().numpy().astype(np.int32)
        bin_d, mc_d, n_present = per_case_metrics(out["pred"], out["seg"])

        ax = axes[row_idx]
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title(f"{label}\nScoliosis_208 (image, downsampled to 512×256)")
        ax[0].axis("off")
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        gt_ids = sorted(set(np.unique(gt).tolist()) - {0})
        ax[1].set_title(f"GT (Jorge): {n_present} vertebrae present\nIDs {gt_ids}")
        ax[1].axis("off")
        ax[2].imshow(img, cmap="gray")
        ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        pred_ids = sorted(set(np.unique(pred).tolist()) - {0})
        ax[2].set_title(f"Pred — bin Dice={bin_d:.3f} | mc Dice={mc_d:.3f}\nPredicted IDs: {pred_ids}")
        ax[2].axis("off")
        diff = np.zeros((*img.shape, 3))
        pred_fg = pred > 0
        gt_fg = gt > 0
        diff[gt_fg & pred_fg] = [0, 1, 0]
        diff[pred_fg & ~gt_fg] = [1, 0, 0]
        diff[gt_fg & ~pred_fg] = [0, 0, 1]
        ax[3].imshow(img, cmap="gray")
        ax[3].imshow(diff, alpha=0.5)
        ax[3].set_title("diff (G=match, R=FP, B=FN)")
        ax[3].axis("off")

    fig.suptitle("Scoliosis_208 zero-shot collapse — fold0 & D1 fail (~0.18-0.23 mc Dice), D2 fits via memorization",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    out_dir = Path("/tmp/scoliosis_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s208_zero_shot.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
