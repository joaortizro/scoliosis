"""Pseudo-label Roboflow X-rays with the Phase 1.2 5-fold ensemble.

Path A — Step 1 in the pseudo-labeling pipeline to break the 213-case
data ceiling and push 5-fold Dice from 0.6946 toward the user target 0.75.

## Pipeline

```
Roboflow image (variable resolution, JPG)
  → preprocess (resize 512×256, /255, mode L)
  → 5 Phase 1.2 fold checkpoints, hflip-TTA per fold
  → average softmax across 5 folds                           # ensemble
  → argmax → predicted multiclass mask (0..17)
  → per-pixel max confidence
  → quality filter (≥14 distinct vertebrae, mean conf ≥ 0.70,
                    foreground fraction in 0.005..0.40)
  → save accepted mask as PNG (matches v2 LabelMultiClass_ID_PNG format)
  → write stats JSON
```

Output structure:

```
data/processed/roboflow_pseudo_labels/
├── images/<stem>.jpg                  (symlink to data/raw/roboflow_scoliosis_v16/...)
├── masks/<stem>.png                   (uint8, values 0..17, accepted only)
├── confidence/<stem>.npy              (float32 per-pixel max-prob, accepted only)
├── stats.json                         (kept/rejected counts, per-rejection-reason breakdown)
└── manifest.csv                       (one row per Roboflow image: stem, accepted, reject_reason, n_vertebrae, mean_confidence, fg_frac)
```

The accepted subset is what Phase 1.4 self-training will consume.

## Quality thresholds (locked in tests/test_pseudo_label_filter.py)

- MIN_VERTEBRAE_FOR_PSEUDO_LABEL = 14    # match v2 trainable coverage profile
- MIN_MEAN_CONFIDENCE = 0.70             # foreground pixels must average ≥ this softmax prob
- MIN_FG_FRAC = 0.005                    # > 0.5% of pixels predicted as vertebra
- MAX_FG_FRAC = 0.40                     # < 40% — rejects "all foreground" failure mode

## Usage

```bash
# Smoke (10 images, no ensemble — single fold)
python scripts/pseudo_label_roboflow.py --smoke

# Full ensemble on all 1535 Roboflow train images
python scripts/pseudo_label_roboflow.py

# Subset by split
python scripts/pseudo_label_roboflow.py --split valid

# Use a different ensemble (e.g., single best fold)
python scripts/pseudo_label_roboflow.py --folds 4
```
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai.inference.predictor import Predictor
from ai.preprocessing.segmentation import NUM_SEG_CLASSES
from ai.training.dataset import IMG_H, IMG_W
from ai.utils import get_device


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_ROOT = REPO_ROOT / "data" / "raw" / "roboflow_scoliosis_v16"
PHASE12_SENTINEL = REPO_ROOT / "experiments" / "results" / "phase1_2_5fold.json"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "roboflow_pseudo_labels"

# Quality thresholds (locked by tests/test_pseudo_label_filter.py)
MIN_VERTEBRAE_FOR_PSEUDO_LABEL: int = 14
MIN_MEAN_CONFIDENCE: float = 0.70
MIN_FG_FRAC: float = 0.005
MAX_FG_FRAC: float = 0.40

NUM_VERTEBRA_CLASSES: int = 17   # T1..L5


def _load_fold_predictors(fold_indices: list[int], device: torch.device) -> list[Predictor]:
    """Load Predictor for each requested fold's run dir from phase1_2_5fold.json."""
    sentinel = json.loads(PHASE12_SENTINEL.read_text())
    folds_by_idx = {f["fold"]: f for f in sentinel["folds"]}
    preds: list[Predictor] = []
    for idx in fold_indices:
        if idx not in folds_by_idx:
            raise ValueError(f"fold {idx} not in sentinel; available: {sorted(folds_by_idx)}")
        run_dir = REPO_ROOT / folds_by_idx[idx]["run_dir"]
        preds.append(Predictor(run_dir, device=device))
        print(f"  loaded fold {idx} from {run_dir.name} (best Dice = {folds_by_idx[idx]['best_val_dice']:.4f})")
    return preds


def _preprocess_image_for_inference(jpg_path: Path) -> torch.Tensor:
    """Load JPG → grayscale → resize 512×256 → /255 → (1, H, W) tensor."""
    pil = Image.open(jpg_path).convert("L")
    pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0   # (H, W)
    return torch.from_numpy(arr).unsqueeze(0)        # (1, H, W)


@torch.no_grad()
def ensemble_predict(
    predictors: list[Predictor],
    image_tensor: torch.Tensor,
    tta: str = "hflip",
) -> tuple[np.ndarray, np.ndarray]:
    """Run each predictor, average softmax, return (mask, confidence_map).

    Args:
        predictors: list of Predictor instances.
        image_tensor: (1, H, W) float in [0, 1].
        tta: "off" or "hflip" passed to each Predictor.

    Returns:
        mask: (H, W) uint8 in 0..17 (argmax across averaged softmax).
        confidence_map: (H, W) float32 — softmax probability of the
            predicted class at each pixel (= max across classes).
    """
    probs_acc: torch.Tensor | None = None
    for p in predictors:
        probs = p.predict_logits(image_tensor, tta=tta).detach().cpu()  # (C, H, W)
        probs_acc = probs if probs_acc is None else (probs_acc + probs)
    assert probs_acc is not None
    probs_mean = probs_acc / len(predictors)
    confidence_map = probs_mean.max(dim=0).values.numpy().astype(np.float32)
    mask = probs_mean.argmax(dim=0).numpy().astype(np.uint8)
    return mask, confidence_map


def pseudo_label_passes_quality(pred: dict) -> tuple[bool, str]:
    """Evaluate quality thresholds. Returns (accepted, reason_if_rejected).

    Args:
        pred: dict with keys 'pred_mask' (H, W uint8) and 'confidence_map' (H, W float).
    """
    mask = pred["pred_mask"]
    conf = pred["confidence_map"]
    n_pixels = mask.size

    # Distinct nonzero classes — proxy for # vertebrae visible
    nonzero_classes = set(np.unique(mask).tolist()) - {0}
    n_vertebrae = len(nonzero_classes)
    if n_vertebrae < MIN_VERTEBRAE_FOR_PSEUDO_LABEL:
        return False, f"only {n_vertebrae} distinct vertebrae predicted (need >= {MIN_VERTEBRAE_FOR_PSEUDO_LABEL})"

    fg_mask = mask > 0
    fg_count = int(fg_mask.sum())
    fg_frac = fg_count / n_pixels
    if fg_frac < MIN_FG_FRAC:
        return False, f"fg fraction {fg_frac:.4f} below {MIN_FG_FRAC}"
    if fg_frac > MAX_FG_FRAC:
        return False, f"fg fraction {fg_frac:.4f} above {MAX_FG_FRAC}"

    if fg_count == 0:
        return False, "no foreground pixels"
    mean_fg_conf = float(conf[fg_mask].mean())
    if mean_fg_conf < MIN_MEAN_CONFIDENCE:
        return False, f"mean fg confidence {mean_fg_conf:.3f} below {MIN_MEAN_CONFIDENCE}"

    return True, ""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "valid", "test", "all"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="which Phase 1.2 folds to ensemble (default: all 5)")
    ap.add_argument("--tta", default="hflip", choices=["off", "hflip"])
    ap.add_argument("--smoke", action="store_true",
                    help="process only 10 images to validate the pipeline")
    args = ap.parse_args()

    print(f"=== pseudo-label Roboflow ===")
    print(f"split={args.split}  folds={args.folds}  tta={args.tta}  smoke={args.smoke}")

    auto_device = get_device()
    if auto_device.backend == "directml":
        # DirectML doesn't support torch.load(map_location=device); fall back to CPU.
        # 1535 imgs × 5-fold ensemble × hflip-TTA on CPU is slow (~30-60 min) but tractable.
        device = torch.device("cpu")
        print(f"device: CPU (DirectML detected but unsupported by torch.load; using CPU)")
    else:
        device = auto_device.device
        print(f"device: {auto_device.name} ({auto_device.backend})")

    print("\n=== loading predictors ===")
    predictors = _load_fold_predictors(args.folds, device=device)

    # Output dirs
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    img_out = OUT_ROOT / "images"
    mask_out = OUT_ROOT / "masks"
    conf_out = OUT_ROOT / "confidence"
    for d in (img_out, mask_out, conf_out):
        _ensure_dir(d)

    splits = [args.split] if args.split != "all" else ["train", "valid", "test"]
    image_paths: list[tuple[str, Path]] = []
    for s in splits:
        sd = ROBOFLOW_ROOT / "images" / s
        if sd.exists():
            for p in sorted(sd.glob("*.jpg")):
                image_paths.append((s, p))

    if args.smoke:
        image_paths = image_paths[:10]

    print(f"\n=== processing {len(image_paths)} images ===")

    rows: list[dict] = []
    rejection_counter: Counter[str] = Counter()
    n_accepted = 0
    for i, (split, img_path) in enumerate(image_paths):
        stem = img_path.stem
        try:
            image_tensor = _preprocess_image_for_inference(img_path)
            mask, conf = ensemble_predict(predictors, image_tensor, tta=args.tta)
            pred = {"pred_mask": mask, "confidence_map": conf}
            accepted, reason = pseudo_label_passes_quality(pred)
        except Exception as e:
            accepted = False
            reason = f"exception: {type(e).__name__}: {e}"
            mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            conf = np.zeros((IMG_H, IMG_W), dtype=np.float32)

        nonzero_classes = set(np.unique(mask).tolist()) - {0}
        fg_mask = mask > 0
        fg_count = int(fg_mask.sum())
        fg_frac = fg_count / mask.size if mask.size else 0.0
        mean_fg_conf = float(conf[fg_mask].mean()) if fg_count else 0.0

        rows.append({
            "stem": stem,
            "split": split,
            "accepted": accepted,
            "reject_reason": reason,
            "n_vertebrae": len(nonzero_classes),
            "fg_frac": fg_frac,
            "mean_fg_confidence": mean_fg_conf,
        })

        if accepted:
            n_accepted += 1
            Image.fromarray(mask).save(mask_out / f"{stem}.png")
            np.save(conf_out / f"{stem}.npy", conf)
            _safe_symlink(img_path, img_out / img_path.name)
        else:
            rejection_counter[reason.split(" ")[0]] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i + 1}/{len(image_paths)}] accepted={n_accepted}  reject_breakdown={dict(rejection_counter.most_common(5))}")

    manifest_path = OUT_ROOT / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    stats = {
        "n_total": len(image_paths),
        "n_accepted": n_accepted,
        "n_rejected": len(image_paths) - n_accepted,
        "acceptance_rate": n_accepted / max(1, len(image_paths)),
        "rejection_breakdown": dict(rejection_counter),
        "thresholds": {
            "min_vertebrae": MIN_VERTEBRAE_FOR_PSEUDO_LABEL,
            "min_mean_confidence": MIN_MEAN_CONFIDENCE,
            "min_fg_frac": MIN_FG_FRAC,
            "max_fg_frac": MAX_FG_FRAC,
        },
        "folds_ensembled": args.folds,
        "tta": args.tta,
        "split": args.split,
        "smoke": args.smoke,
    }
    stats_path = OUT_ROOT / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n=== done ===")
    print(json.dumps(stats, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"stats:    {stats_path}")


if __name__ == "__main__":
    main()
