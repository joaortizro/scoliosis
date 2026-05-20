"""Figura S_208 para el paper IEEE — modo de falla de asignación de identificadores.

Renderiza una grilla 2 x 4:
  Fila (a) Phase 1.2 fold-0 (zero-shot sobre S_208, nunca lo vio en entrenamiento)
  Fila (b) D2 (vio S_208 durante entrenamiento — base de memorización)

Columnas: radiografía | GT | predicción | mapa de diferencia binaria.

Output: docs/figures/s208_paper.png (carpeta gitignored — imágenes con datos
de paciente no se versionan; render reproducible vía este script + checkpoints
trazados por DVC).
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
    # IEEE conference single column ≈ 88 mm = 3.46 in. Render a 3.4 in figure
    # so that \includegraphics[width=\linewidth] uses 1:1 scaling and the in-figure
    # fonts land at ~7--8 pt on the printed page.
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "font.family": "serif",
    })

    models = [
        ("(a) Zero-shot (IBIO-SD)",
         "ai/models/checkpoints/encoder_unet/20260509_194823_b41714d16d325371"),
        ("(b) D2 (IBIO-SD + ERS-18)",
         "ai/models/checkpoints/encoder_unet/20260517_050245_1b8fe848ffa7b4fb"),
    ]

    rows_csv = pd.read_csv("data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv")
    s208 = rows_csv[(rows_csv["category"] == "Scoliosis") & (rows_csv["patient_id"] == 208)].iloc[0]

    cmap = vertebra_cmap()
    fig, axes = plt.subplots(len(models), 4, figsize=(3.4, 4.6),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.34})

    col_titles = ["Radiografía", "GT", "Predicción", "Diferencia"]

    for row_idx, (panel_label, run_dir) in enumerate(models):
        predictor = Predictor(run_dir, device=torch.device("cpu"))
        out = predictor.predict_from_row(s208, tta="off")
        img = out["image"].squeeze().cpu().numpy()
        gt = out["seg"].cpu().numpy().astype(np.int32)
        pred = out["pred"].cpu().numpy().astype(np.int32)
        bin_d, mc_d = per_case_metrics(out["pred"], out["seg"])

        ax = axes[row_idx]

        for k in range(4):
            ax[k].set_xticks([]); ax[k].set_yticks([])
            if row_idx == 0:
                ax[k].set_title(col_titles[k], fontsize=7, pad=2)

        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=17, alpha=0.55)
        ax[2].imshow(img, cmap="gray")
        ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=17, alpha=0.55)

        diff = np.zeros((*img.shape, 3))
        pred_fg = pred > 0; gt_fg = gt > 0
        diff[gt_fg & pred_fg] = [0, 0.8, 0]
        diff[pred_fg & ~gt_fg] = [0.9, 0, 0]
        diff[gt_fg & ~pred_fg] = [0, 0, 0.9]
        ax[3].imshow(img, cmap="gray")
        ax[3].imshow(diff, alpha=0.5)

        # Banner por fila arriba de la cuadrícula (panel_label + métricas)
        banner = (
            rf"{panel_label}   $\bullet$   "
            rf"Dice$_{{\mathrm{{bin}}}}={bin_d:.2f}$, "
            rf"Dice$_{{\mathrm{{mc}}}}={mc_d:.2f}$"
        )
        bbox0 = ax[0].get_position()
        bbox3 = ax[3].get_position()
        y_banner = bbox0.y1 + (0.040 if row_idx == 0 else 0.020)
        fig.text((bbox0.x0 + bbox3.x1) / 2, y_banner, banner,
                  ha="center", va="bottom", fontsize=7, weight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.01)
    out_dir = REPO_ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s208_paper.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"escrito: {out_path}")


if __name__ == "__main__":
    main()
