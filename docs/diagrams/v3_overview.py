"""Render architecture + pipeline + metrics overview for Model Primer v3."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")

ENCODER_COLOR = "#4C72B0"
DECODER_COLOR = "#55A868"
SKIP_COLOR = "#C44E52"
DATA_COLOR = "#8172B2"
METRIC_COLOR = "#DD8452"


def box(ax, xy, w, h, text, color, fontsize=9, alpha=0.85, text_color="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="black", facecolor=color, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight="bold")


def arrow(ax, start, end, color="black", style="-|>", lw=1.2, curve=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={curve}",
    ))


def panel_architecture(ax):
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.set_title("1 · EncoderUNet architecture  (ResNet-34 + U-Net decoder, ~24M params)",
                 fontsize=11, weight="bold", loc="left")

    # Input
    box(ax, (0.2, 2.6), 1.3, 0.9, "Input\n1×512×256\ngrayscale", DATA_COLOR, fontsize=8)

    # Encoder stages
    enc_xs = [1.9, 3.2, 4.5, 5.8, 7.1]
    enc_labels = [
        "conv1+pool\n64",
        "stage1\n64",
        "stage2\n128",
        "stage3\n256",
        "stage4\n512",
    ]
    enc_h = [1.8, 1.6, 1.4, 1.2, 1.0]
    for x, lab, h in zip(enc_xs, enc_labels, enc_h):
        y = 3.05 - (1.8 - h) / 2
        box(ax, (x, y), 1.1, h, lab, ENCODER_COLOR, fontsize=7.5)

    # Bottleneck arrow region
    arrow(ax, (1.5, 3.05), (1.9, 3.05))
    for i in range(len(enc_xs) - 1):
        arrow(ax, (enc_xs[i] + 1.1, 3.05), (enc_xs[i + 1], 3.05))

    # Decoder stages (mirror on the right, lower row)
    dec_xs = [7.1, 5.8, 4.5, 3.2]
    dec_labels = [
        "up1\n256",
        "up2\n128",
        "up3\n64",
        "up4\n32",
    ]
    for x, lab in zip(dec_xs, dec_labels):
        box(ax, (x, 0.6), 1.1, 1.0, lab, DECODER_COLOR, fontsize=7.5)
    # Decoder flow right→left at y≈1.1
    arrow(ax, (7.1 + 0.55, 2.55), (7.1 + 0.55, 1.6))  # bottleneck→up1
    for i in range(len(dec_xs) - 1):
        arrow(ax, (dec_xs[i], 1.1), (dec_xs[i + 1] + 1.1, 1.1))

    # Skip connections (encoder top → decoder top)
    skips = [
        (enc_xs[0] + 0.55, 3.05 + 0.9, dec_xs[3] + 0.55, 1.6),
        (enc_xs[1] + 0.55, 3.05 + 0.8, dec_xs[2] + 0.55, 1.6),
        (enc_xs[2] + 0.55, 3.05 + 0.7, dec_xs[1] + 0.55, 1.6),
        (enc_xs[3] + 0.55, 3.05 + 0.6, dec_xs[0] + 0.55, 1.6),
    ]
    for x1, y1, x2, y2 in skips:
        arrow(ax, (x1, y1), (x2, y2), color=SKIP_COLOR, lw=1.0, curve=-0.35)

    # Seg head + output
    box(ax, (1.9, 0.6), 1.1, 1.0, "seg head\n1×1 conv", DECODER_COLOR, fontsize=7.5)
    arrow(ax, (dec_xs[-1], 1.1), (3.0, 1.1))
    box(ax, (0.2, 0.6), 1.3, 1.0, "Seg logits\n17×512×256\n→ argmax", DATA_COLOR, fontsize=8)
    arrow(ax, (1.9, 1.1), (1.5, 1.1))

    # Legend
    ax.text(11.8, 5.5, "Skip connections", color=SKIP_COLOR, fontsize=8, weight="bold")
    ax.text(11.8, 5.1, "Encoder (frozen 10ep warmup, lr 1e-4)",
            color=ENCODER_COLOR, fontsize=8, weight="bold")
    ax.text(11.8, 4.7, "Decoder (trained from scratch, lr 1e-3)",
            color=DECODER_COLOR, fontsize=8, weight="bold")
    ax.text(11.8, 4.3, "Data tensor", color=DATA_COLOR, fontsize=8, weight="bold")

    ax.text(11.8, 3.3, "Classes (17):\nbg + T1..T12 + L1..L5",
            fontsize=8, color="black")
    ax.text(11.8, 2.2, "Grayscale stem:\nRGB conv1 avg → 1ch",
            fontsize=8, color="black")
    ax.text(11.8, 1.2, "Aug v4 (train only):\n±15° rot, translate, elastic,\ngamma, CLAHE, blur, noise",
            fontsize=8, color="black")

    ax.set_axis_off()


def panel_cobb(ax):
    ax.set_xlim(0, 14); ax.set_ylim(0, 4)
    ax.set_title("2 · Cobb angle pipeline  (smoothed tangent method, window=3)",
                 fontsize=11, weight="bold", loc="left")

    stages = [
        (0.2, "Seg mask\n17 classes"),
        (2.3, "Per-vertebra\ncentroids\n(cx, cy)"),
        (4.5, "Smooth curve\nalong spine\n(moving avg)"),
        (7.0, "Tangent slopes\nat inflection points\nabove & below apex"),
        (9.8, "Angle between\ntangents\n= Cobb°"),
        (12.3, "Severity bucket\nnormal/mild/\nmoderate/severe"),
    ]
    for x, lab in stages:
        box(ax, (x, 1.4), 1.9, 1.3, lab, METRIC_COLOR, fontsize=8)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.9
        x2 = stages[i + 1][0]
        arrow(ax, (x1, 2.05), (x2, 2.05), lw=1.4)

    ax.text(0.2, 0.3,
            "Pipeline floor (GT mask → tangent): MAE ≈ 10.4°, r ≈ 0.89.  "
            "Model contributes ~0–1° extra on top of the geometry floor.",
            fontsize=9, color="black", style="italic")
    ax.set_axis_off()


def panel_metrics(ax):
    ax.set_xlim(0, 14); ax.set_ylim(0, 4)
    ax.set_title("3 · Metrics & evaluation protocol",
                 fontsize=11, weight="bold", loc="left")

    # Segmentation metrics
    ax.text(0.2, 3.4, "Segmentation (per-class, mean over 17 vertebrae)",
            fontsize=9.5, weight="bold")
    ax.text(0.2, 2.9,
            r"Dice = $\dfrac{2\,|P\cap G|}{|P|+|G|}$     "
            r"IoU = $\dfrac{|P\cap G|}{|P\cup G|}$     "
            r"HD95 = 95-th pct symmetric Hausdorff (px)",
            fontsize=9)

    # Cobb metrics
    ax.text(0.2, 2.2, "Cobb angle (regression)", fontsize=9.5, weight="bold")
    ax.text(0.2, 1.75,
            r"MAE = mean $|\hat\theta - \theta|$      "
            r"SMAPE = $|\hat\theta - \theta| \,/\, \frac{|\hat\theta|+|\theta|}{2}$      "
            r"Pearson r on $(\hat\theta, \theta)$",
            fontsize=9)

    # Severity
    ax.text(0.2, 1.1, "Severity (4-class: normal <10° · mild 10–25° · moderate 25–40° · severe ≥40°)",
            fontsize=9.5, weight="bold")
    ax.text(0.2, 0.65,
            r"Accuracy · macro-F1 · Cohen's κ · confusion matrix",
            fontsize=9)

    # Protocol box (right side)
    box(ax, (9.6, 0.4), 4.2, 3.2,
        "Protocol\n\n"
        "• Single 80/20 split\n   (152 trainable cases)\n"
        "• 5-fold CV for final\n   comparison (mean ± std)\n"
        "• TTA at inference:\n   flip H + avg logits\n"
        "• Pipeline floor check:\n   GT mask → same Cobb fn",
        "#333333", fontsize=9, alpha=0.92)

    ax.set_axis_off()


def main() -> Path:
    fig = plt.figure(figsize=(15.5, 12.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.95, 1.0], hspace=0.35)

    panel_architecture(fig.add_subplot(gs[0]))
    panel_cobb(fig.add_subplot(gs[1]))
    panel_metrics(fig.add_subplot(gs[2]))

    fig.suptitle("Model Primer v3 — Scoliosis segmentation + Cobb angle pipeline",
                 fontsize=13, weight="bold", y=0.995)

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}")
