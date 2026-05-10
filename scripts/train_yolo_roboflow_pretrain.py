"""Phase 3b.2c — YOLOv8-Pose pretrain on filtered Roboflow.

Spec: docs/superpowers/specs/2026-05-10-path-b-detection-cobb-design.md §8.

Pretrains a YOLOv8-Pose detector on the Roboflow scoliosis dataset
(filtered to ≥14 vertebrae per image, dummy keypoints with visibility=0
so kpt loss is masked — bbox-only pretraining).

Output:
    ai/models/checkpoints/yolo_vertebra/<TIMESTAMP>_pretrain/
        ├── cfg.json
        ├── results.csv
        ├── weights/{best,last}.pt
        └── ...
    experiments/results/yolo_roboflow_pretrain.json   (sentinel)

Determinism (spec §10.1):
    PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8
    deterministic=True, seed=42

Gate (spec §8): mAP@0.5 ≥ 0.85 on Roboflow valid.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "processed" / "yolo_pose_datasets" / "roboflow_pretrain" / "data.yaml"
RUN_ROOT = REPO_ROOT / "ai" / "models" / "checkpoints" / "yolo_vertebra"
SENTINEL_PATH = REPO_ROOT / "experiments" / "results" / "yolo_roboflow_pretrain.json"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 512
DEFAULT_PATIENCE = 10
MAP05_GATE = 0.85


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="5 epochs smoke test")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 5
        args.patience = 3

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Defer ultralytics import — large + GPU-bound, makes script-loading slow in tests
    from ultralytics import YOLO
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_pretrain"
    project_dir = RUN_ROOT
    project_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "phase": "3b.2c",
        "model": "yolov8n-pose.pt",
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "device": args.device,
        "seed": 42,
        "deterministic": True,
        "kpt_loss_masked": True,
        "started_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    print("=== Phase 3b.2c YOLOv8-Pose Roboflow pretrain ===")
    print(json.dumps(cfg, indent=2))

    model = YOLO("yolov8n-pose.pt")  # pretrained on COCO keypoints
    start = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        seed=42,
        deterministic=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        # Augmentation defaults from ultralytics
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        # Loss weights — leave at ultralytics defaults
        # kpt loss is masked automatically by visibility=0 (no flag needed)
    )
    wall = time.time() - start

    # Validate on Roboflow valid + test for completeness
    metrics_valid = model.val(data=str(DATA_YAML), split="val", device=args.device)
    metrics_test = model.val(data=str(DATA_YAML), split="test", device=args.device)

    map50_valid = float(metrics_valid.box.map50)
    map5095_valid = float(metrics_valid.box.map)
    map50_test = float(metrics_test.box.map50)
    map5095_test = float(metrics_test.box.map)
    gate_passed = map50_valid >= MAP05_GATE

    sentinel = {
        **cfg,
        "wall_seconds": wall,
        "wall_minutes": wall / 60,
        "valid_map50": map50_valid,
        "valid_map5095": map5095_valid,
        "test_map50": map50_test,
        "test_map5095": map5095_test,
        "map50_gate": MAP05_GATE,
        "gate_passed": gate_passed,
        "run_dir": str(project_dir / run_name),
        "best_weights": str(project_dir / run_name / "weights" / "best.pt"),
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL_PATH.write_text(json.dumps(sentinel, indent=2))
    (project_dir / run_name / "cfg.json").write_text(json.dumps(sentinel, indent=2))

    print("\n=== Results ===")
    print(f"valid mAP@0.5 = {map50_valid:.4f}  mAP@0.5:0.95 = {map5095_valid:.4f}")
    print(f"test  mAP@0.5 = {map50_test:.4f}  mAP@0.5:0.95 = {map5095_test:.4f}")
    print(f"wall = {wall/60:.1f} min")
    print(f"gate ≥ {MAP05_GATE}: {'PASS' if gate_passed else 'FAIL'}")
    print(f"sentinel: {SENTINEL_PATH}")


if __name__ == "__main__":
    main()
