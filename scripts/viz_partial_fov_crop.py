"""Visualize gentle vs aggressive RandomVerticalCrop samples.

Pulls 3 cases from the clean index, runs them through the same
``preprocess_case`` (with ``roi_crop_mode="from_mask"`` so the spine
fills the frame), and renders one row per case:

    | original | gentle sample | aggressive sample | aggressive worst-case |

Every cropped subplot is labelled with the sampled ``f`` and ``mode``.
The mask is overlaid with a semi-transparent colormap so the centroid
policy (vertebrae outside the window are removed) is visible at a
glance. Used as a sanity-check artifact for
[[2026-05-15_partial_fov_experiment_plan]].

Run:
    python scripts/viz_partial_fov_crop.py --out experiments/viz/partial_fov_crop_samples.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from ai.preprocessing.transforms import deterministic_vertical_crop  # noqa: E402
from ai.training.dataset import preprocess_case  # noqa: E402

log = logging.getLogger(__name__)

CLEAN_INDEX = REPO / "data/processed/audit_v2_corrected/clean_index.csv"


def _pick_cases(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Pick a diverse trio: one Normal + two Scoliosis at different Cobb."""
    ok = df[(df["status"] == "ok") & (df["target_vertebrae_count"] == 17)].copy()
    normal = ok[ok["category"] == "Normal"].head(1)
    scoli = ok[ok["category"] == "Scoliosis"].copy()
    scoli = scoli.sort_values("cobb_angle_deg")
    mild = scoli.head(1)
    severe = scoli.tail(1)
    picks = pd.concat([normal, mild, severe], ignore_index=True)
    return picks.head(n)


def _overlay(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Render grayscale image + multiclass mask overlay (jet @ alpha 0.4)."""
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    masked = np.ma.masked_where(mask == 0, mask)
    n_classes = 17
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    cmap = ListedColormap(colors)
    ax.imshow(masked, cmap=cmap, vmin=1, vmax=n_classes, alpha=0.55)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="notebooks/sandbox/viz_2026-05-15_partial_fov/partial_fov_crop_samples.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clean-index", default=str(CLEAN_INDEX))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(args.clean_index)
    cases = _pick_cases(df, n=3)
    log.info("picked cases:\n%s", cases[["patient_id", "category", "cobb_angle_deg"]].to_string(index=False))

    rng = np.random.default_rng(args.seed)
    f_gentle = float(rng.uniform(0.5, 1.0))
    f_aggressive = float(rng.uniform(0.3, 1.0))

    n_rows = len(cases)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 7.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, (_, row) in enumerate(cases.iterrows()):
        case = preprocess_case(row, clahe_mode="off", roi_crop_mode="from_mask")
        image = case["image"]
        seg = case["seg"]

        case_label = f"{row['category'][0]}_{int(row['patient_id'])}"
        cobb = row.get("cobb_angle_deg")
        cobb_str = f" · Cobb {cobb:.0f}°" if cobb and not np.isnan(cobb) else ""

        # Per-row independent seed so each row uses a different draw.
        row_rng = np.random.default_rng(args.seed + 100 * r)

        # Col 0 — original
        _overlay(
            axes[r, 0],
            image[0].numpy(),
            seg.numpy(),
            f"{case_label}{cobb_str}\noriginal (f=1.0)",
        )

        # Col 1 — gentle (random mode, fixed f from this row's rng)
        gen = np.random.default_rng(args.seed + 100 * r + 1)
        fg = float(gen.uniform(0.5, 1.0))
        gimg, gseg = deterministic_vertical_crop(
            image, seg, f=fg, mode="random", rng=row_rng,
        )
        _overlay(
            axes[r, 1],
            gimg[0].numpy(),
            gseg.numpy(),
            f"gentle (M1a)\nf={fg:.2f}, mode=random",
        )

        # Col 2 — aggressive (random mode, fresh draw)
        gen = np.random.default_rng(args.seed + 100 * r + 2)
        fa = float(gen.uniform(0.3, 1.0))
        aimg, aseg = deterministic_vertical_crop(
            image, seg, f=fa, mode="random", rng=row_rng,
        )
        _overlay(
            axes[r, 2],
            aimg[0].numpy(),
            aseg.numpy(),
            f"aggressive (M1b)\nf={fa:.2f}, mode=random",
        )

        # Col 3 — worst-case demo: f=0.3, mode=top (chest-only film)
        timg, tseg = deterministic_vertical_crop(
            image, seg, f=0.3, mode="top",
        )
        _overlay(
            axes[r, 3],
            timg[0].numpy(),
            tseg.numpy(),
            "worst case\nf=0.30, mode=top (chest film)",
        )

    fig.suptitle(
        "RandomVerticalCrop: gentle (M1a) vs aggressive (M1b)\n"
        f"gentle f∈[0.5, 1.0]   ·   aggressive f∈[0.3, 1.0]   ·   seed={args.seed}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    log.info("wrote %s", out_path)

    # Also emit one PNG per case (single-row figure) so each is
    # legible when viewed in isolation.
    per_case_dir = out_path.parent
    for r, (_, row) in enumerate(cases.iterrows()):
        sub_fig, sub_axes = plt.subplots(1, 4, figsize=(20, 8))
        for c in range(4):
            src_ax = axes[r, c]
            sub_axes[c].imshow(src_ax.images[0].get_array(), cmap="gray", vmin=0.0, vmax=1.0)
            if len(src_ax.images) > 1:
                sub_axes[c].imshow(
                    src_ax.images[1].get_array(),
                    cmap=src_ax.images[1].get_cmap(),
                    alpha=0.55, vmin=1, vmax=17,
                )
            sub_axes[c].set_title(src_ax.get_title(), fontsize=15)
            sub_axes[c].set_xticks([])
            sub_axes[c].set_yticks([])
        case_label = f"{row['category'][0]}_{int(row['patient_id'])}"
        sub_fig.suptitle(f"Case {case_label}  ·  RandomVerticalCrop samples", fontsize=17)
        sub_fig.tight_layout(rect=[0, 0, 1, 0.94])
        sub_path = per_case_dir / f"partial_fov_crop_{case_label}.png"
        sub_fig.savefig(sub_path, dpi=140, bbox_inches="tight")
        plt.close(sub_fig)
        log.info("wrote %s", sub_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
