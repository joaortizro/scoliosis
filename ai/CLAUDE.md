# ai/ — Scoliosis ML Package

## Purpose
Installable PyTorch library (`scoliosis-ai`) for spine segmentation, keypoint extraction, and Cobb angle measurement. Pure ML — no server, no hexagonal layers.

## Entry Points
- `preprocessing/segmentation.py` — mask remapping to training labels
- `preprocessing/keypoints.py` — vertebra corner extraction from masks
- `evaluation/cobb.py` — Cobb angle derivation (centroid-polynomial and endplate-tilt methods)
- `utils/device.py` — cross-backend device detection (CUDA / ROCm / DirectML / CPU)

## Key Files

| File | Responsibility |
|------|---------------|
| `preprocessing/segmentation.py` | `remap_to_target_classes()`, `NUM_SEG_CLASSES` (18) |
| `preprocessing/keypoints.py` | `multiclass_mask_to_keypoints()` → (68, 2) array |
| `evaluation/cobb.py` | `cobb_from_segmentation()`, `cobb_from_keypoints()` |
| `utils/device.py` | `get_device()` → `DeviceInfo` dataclass |
| `models/architectures/base_model.py` | Abstract `BaseModel(nn.Module)` |
| `training/trainer.py` | Training orchestrator (stub — real loops live in notebooks/scripts) |
| `inference/predictor.py` | Checkpoint loading + inference (stub) |

## Conventions
- All hyperparameters received as function args — never read `params.yaml` directly
- `target_ids` kwarg controls which vertebra IDs to process (v1 default: 6..22, v2: 1..17)
- Masks are 2D `np.ndarray`, background=0, classes=1..N
- Keypoints use NaN for missing vertebrae, never zero-padding
- Device placement via `.to(device)`, never `.cuda()` — DirectML compatibility required
- New architectures get their own module under `models/`; never modify existing arch files

## Cross-References
- See `preprocessing/CLAUDE.md` for mask/keypoint pipeline details
- See `evaluation/CLAUDE.md` for Cobb algorithm and metric functions
- See `../server/` for the infrastructure adapter that wraps this package
- See `../scripts/` for DVC entrypoints that call into this package
