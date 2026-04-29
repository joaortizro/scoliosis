# ai/preprocessing/ — Mask Remapping & Keypoint Extraction

## Purpose
Transform raw MaIA dataset masks into training-ready segmentation labels and vertebra corner keypoints.

## Entry Points
- `segmentation.py` — start here for understanding the label pipeline
- `keypoints.py` — start here for endplate corner extraction

## Key Functions

| Function | File | Signature | Returns |
|----------|------|-----------|---------|
| `remap_to_target_classes` | `segmentation.py` | `(mask, target_ids=TARGET_VERTEBRA_IDS)` | `np.ndarray` uint8, values 0..len(target_ids) |
| `multiclass_mask_to_keypoints` | `keypoints.py` | `(mask, target_ids=TARGET_VERTEBRA_IDS)` | `np.ndarray` float (68, 2) — NaN-padded |

## Constants

| Constant | File | Value | Meaning |
|----------|------|-------|---------|
| `TARGET_VERTEBRA_IDS` | both | `tuple(range(6, 23))` | v1 default: T1..L5 raw IDs |
| `NUM_SEG_CLASSES` | `segmentation.py` | `18` | 0=background + 17 vertebrae |
| `KEYPOINTS_PER_VERTEBRA` | `keypoints.py` | `4` | TL, TR, BL, BR corners |
| `TOTAL_KEYPOINTS` | `keypoints.py` | `68` | 17 vertebrae x 4 corners |

## Dataset Version Handling
- **v1 masks** (raw IDs 6..22): call with default `target_ids` or omit the arg
- **v2 masks** (raw IDs 1..17): pass `target_ids=tuple(range(1, 18))`
- After remapping, both produce identical label space: 0=bg, 1..17=T1..L5

## Keypoint Layout
4 corners per vertebra in order: TL, TR, BL, BR. Flattened across 17 vertebrae:
`[TL_T1, TR_T1, BL_T1, BR_T1, TL_T2, ..., BR_L5]` → shape (68, 2) as (x_px, y_px).
Missing vertebrae filled with NaN rows.

## Conventions
- Inputs are always raw ID masks (before remap) — both functions handle the mapping internally
- Corner detection uses PCA on per-vertebra pixel coordinates (`_oriented_corners`)
- Deterministic given same input — no randomness in preprocessing
