"""Phase 3b.3 / 3b.4 — YOLOv8-Pose v2 fine-tune driver (per fold).

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §8.

Fine-tunes a YOLOv8-Pose detector on a single v2 fold (170 train + 43 val
typical) starting from one of:
    --init coco           (yolov8n-pose.pt, COCO-pretrained)
    --init pretrain        (the Roboflow-pretrained best.pt from 3b.2c)
    --init <path>          (any other .pt file)

Backbone freeze: ultralytics `freeze=10` keeps the CSPDarknet backbone
frozen and trains the neck + head only — head-only adaptation per spec
risk-mitigation #2. Set `--freeze 0` to fine-tune the whole model.

Usage:
    python scripts/train_yolo_v2_finetune.py --fold 0 --init coco
    python scripts/train_yolo_v2_finetune.py --fold 0 --init pretrain
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "yolo_pose_datasets"
RUN_ROOT = REPO_ROOT / "ai" / "models" / "checkpoints" / "yolo_vertebra"
PRETRAIN_BEST = RUN_ROOT / "20260510_200816_pretrain" / "weights" / "best.pt"
SENTINEL_DIR = REPO_ROOT / "experiments" / "results"

DEFAULT_EPOCHS = 80
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 512
DEFAULT_PATIENCE = 15
DEFAULT_FREEZE = 10  # CSPDarknet backbone


def _resolve_init(init: str) -> str:
    if init == "coco":
        return "yolov8n-pose.pt"
    if init == "pretrain":
        if not PRETRAIN_BEST.exists():
            raise FileNotFoundError(f"pretrain checkpoint not found at {PRETRAIN_BEST}")
        return str(PRETRAIN_BEST)
    p = Path(init)
    if not p.exists():
        raise FileNotFoundError(f"init checkpoint not found: {init}")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--init", default="coco", help="coco | pretrain | <path-to-pt>")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--freeze", type=int, default=DEFAULT_FREEZE,
                    help="number of leading layers to freeze (10=backbone)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="5 epochs smoke test")
    ap.add_argument("--name-suffix", default="", help="appended to run dir name")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 5
        args.patience = 3

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from ultralytics import YOLO
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    init_path = _resolve_init(args.init)
    init_tag = args.init if args.init in {"coco", "pretrain"} else "custom"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_v2_fold{args.fold}_init-{init_tag}{args.name_suffix}"

    data_yaml = DATASET_ROOT / f"v2_fold_{args.fold}" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"v2 fold dataset missing: {data_yaml}. Run scripts/prepare_yolo_pose_datasets.py first."
        )

    cfg = {
        "phase": "3b.3" if args.smoke else "3b.4-fold",
        "fold": args.fold,
        "init": args.init,
        "init_path": init_path,
        "model": "yolov8n-pose",
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "freeze_layers": args.freeze,
        "device": args.device,
        "seed": 42,
        "deterministic": True,
        "started_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    print("=== v2 fine-tune ===")
    print(json.dumps(cfg, indent=2))

    model = YOLO(init_path)
    start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        seed=42,
        deterministic=True,
        freeze=args.freeze,
        project=str(RUN_ROOT),
        name=run_name,
        exist_ok=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
    )
    wall = time.time() - start

    # Validate on the fold's val split — this gives box + pose mAP
    metrics_val = model.val(data=str(data_yaml), split="val", device=args.device)

    sentinel = {
        **cfg,
        "wall_seconds": wall,
        "wall_minutes": wall / 60,
        "val_box_map50": float(metrics_val.box.map50),
        "val_box_map5095": float(metrics_val.box.map),
        "val_pose_map50": float(metrics_val.pose.map50),
        "val_pose_map5095": float(metrics_val.pose.map),
        "run_dir": str(RUN_ROOT / run_name),
        "best_weights": str(RUN_ROOT / run_name / "weights" / "best.pt"),
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    sentinel_path = SENTINEL_DIR / f"yolo_v2_fold{args.fold}_init-{init_tag}{args.name_suffix}.json"
    sentinel_path.write_text(json.dumps(sentinel, indent=2))
    (RUN_ROOT / run_name / "cfg.json").write_text(json.dumps(sentinel, indent=2))

    print(f"\n=== Results fold {args.fold} init={args.init} ===")
    print(f"box mAP@0.5 = {sentinel['val_box_map50']:.4f}")
    print(f"box mAP@0.5:0.95 = {sentinel['val_box_map5095']:.4f}")
    print(f"pose mAP@0.5 = {sentinel['val_pose_map50']:.4f}")
    print(f"pose mAP@0.5:0.95 = {sentinel['val_pose_map5095']:.4f}")
    print(f"wall = {wall/60:.1f} min")
    print(f"sentinel: {sentinel_path}")


if __name__ == "__main__":
    main()
