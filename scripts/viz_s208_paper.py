"""Publication-quality figure of the Scoliosis_208 ID-assignment failure.

Designed for inclusion as a single-column or 2-column figure in the IEEE
conference paper. Renders a 2-row × 4-col grid:
  Row 1: fold0 (well-trained v2-only, zero-shot)
  Row 2: D2 (trained on this case — memorization)

Columns: input radiograph, GT segmentation, prediction, foreground diff.

Output PNG saved to /tmp/scoliosis_viz/s208_paper.png (gitignored per
patient-data confidentiality rule). User must copy into the paper repo
manually before figure inclusion.
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


def per_case_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred_b = pred.long().unsqueeze(0)
    target_b = target.long().unsqueeze(0)
    c = confusion_per_class(pred_b, target_b, num_classes=NUM_SEG_CLASSES)
    tp, fp, fn = c["tp"].double(), c["fp"].double(), c["fn"].double()
    case_dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    present = (tp + fn) > 0
    fg_present = present[1:]
    macro = float(case_dice[1:][fg_present].mean().item()) if fg_present.any() else float("nan")
    pred_fg = (pred > 0); gt_fg = (target > 0)
    inter = (pred_fg & gt_fg).sum().double()
    union = pred_fg.sum().double() + gt_fg.sum().double()
    binary = float((2 * inter / (union + 1e-9)).item())
    return binary, macro


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "font.family": "serif",
    })

    models = [
        ("(a) Phase 1.2 fold-0 — zero-shot", "v2-only training, never saw this case",
         "ai/models/checkpoints/encoder_unet/20260509_194823_b41714d16d325371"),
        ("(b) D2 — trained on this case", "memorization baseline, n=18 added to train",
         "ai/models/checkpoints/encoder_unet/20260517_050245_1b8fe848ffa7b4fb"),
    ]

    rows_csv = pd.read_csv("data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv")
    s208 = rows_csv[(rows_csv["category"] == "Scoliosis") & (rows_csv["patient_id"] == 208)].iloc[0]

    cmap = vertebra_cmap()
    fig, axes = plt.subplots(len(models), 4, figsize=(11.5, 5.5))

    col_titles = ["Radiograph", "Ground-truth labels", "Model prediction", "Foreground diff"]

    for row_idx, (panel_label, panel_desc, run_dir) in enumerate(models):
        predictor = Predictor(run_dir, device=torch.device("cpu"))
        out = predictor.predict_from_row(s208, tta="off")
        img = out["image"].squeeze().cpu().numpy()
        gt = out["seg"].cpu().numpy().astype(np.int32)
        pred = out["pred"].cpu().numpy().astype(np.int32)
        bin_d, mc_d = per_case_metrics(out["pred"], out["seg"])

        ax = axes[row_idx]
        ax[0].imshow(img, cmap="gray")
        if row_idx == 0:
            ax[0].set_title(col_titles[0])
        ax[0].set_ylabel(f"{panel_label}\n{panel_desc}", fontsize=10, rotation=0,
                          labelpad=110, ha="right", va="center")
        ax[0].set_xticks([]); ax[0].set_yticks([])

        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        if row_idx == 0:
            ax[1].set_title(col_titles[1])
        ax[1].set_xticks([]); ax[1].set_yticks([])

        ax[2].imshow(img, cmap="gray")
        ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        if row_idx == 0:
            ax[2].set_title(col_titles[2])
        ax[2].text(0.5, -0.06,
                    f"binary Dice = {bin_d:.2f}  |  macro mc Dice = {mc_d:.2f}",
                    transform=ax[2].transAxes, ha="center", fontsize=10)
        ax[2].set_xticks([]); ax[2].set_yticks([])

        diff = np.zeros((*img.shape, 3))
        pred_fg = pred > 0; gt_fg = gt > 0
        diff[gt_fg & pred_fg] = [0, 0.8, 0]
        diff[pred_fg & ~gt_fg] = [0.9, 0, 0]
        diff[gt_fg & ~pred_fg] = [0, 0, 0.9]
        ax[3].imshow(img, cmap="gray")
        ax[3].imshow(diff, alpha=0.5)
        if row_idx == 0:
            ax[3].set_title(col_titles[3])
        ax[3].text(0.5, -0.06,
                    "green: TP foreground  |  red: FP  |  blue: FN",
                    transform=ax[3].transAxes, ha="center", fontsize=9)
        ax[3].set_xticks([]); ax[3].set_yticks([])

    fig.suptitle("Failure-mode analysis: ID-assignment fragility under extreme scoliosis (Scoliosis_208, OOD source)",
                  fontsize=12, y=1.02)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    out_dir = Path("/tmp/scoliosis_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s208_paper.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
