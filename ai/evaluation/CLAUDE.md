# ai/evaluation/ — Cobb Angle & Metrics

## Purpose
Derive Cobb angles from model predictions (segmentation masks or keypoints) and compute evaluation metrics.

## Entry Points
- `cobb.py` — Cobb angle computation (three algorithms)
- `evaluator.py` — metric aggregation (stub — full eval logic lives in notebooks for now)

## Cobb Angle Functions

| Function | Input | Output | Algorithm |
|----------|-------|--------|-----------|
| `cobb_from_segmentation(mask)` | 2D label map (0..17) | degrees [0, 90] | Centroid polynomial fit |
| `cobb_from_segmentation_tangent(mask)` | 2D label map (0..17) | degrees >= 0 | Smoothed centroid tangent |
| `cobb_from_keypoints(keypoints)` | (68, 2) float array | degrees [0, 90] | Centroid polynomial fit |
| `cobb_from_segmentation_endplates(mask)` | 2D label map (0..17) | degrees [0, 180] | Endplate tilt variant |
| `cobb_from_raw_multiclass_mask(mask, target_ids)` | Raw ID mask | degrees [0, 90] | Polynomial convenience |
| `cobb_from_raw_multiclass_mask_tangent(mask, target_ids)` | Raw ID mask | degrees >= 0 | Tangent convenience |

## Algorithm: Centroid Polynomial Fit (original)
1. Compute centroid (x, y) of each vertebra class present in the mask
2. Fit degree-5 polynomial `x = p(y)` through centroids
3. Sample 400 dense points along y-axis, compute `dx/dy` slopes
4. Cobb angle = arctan difference between max and min slopes

## Algorithm: Smoothed Centroid Tangent (recommended for model eval)
1. Compute centroid (x, y) of each vertebra class, sort top→bottom
2. Compute piecewise dx, dy between consecutive centroids
3. Smooth with running average (window=3)
4. Cobb angle = range of arctan2(dx, dy) angles
5. Better at training resolution: MAE 17.19° vs 19.65° polynomial on 179 GT cases at 512x256

## Constants
- `NUM_TARGET_VERTEBRAE = 17`
- `_POLY_DEGREE = 5` — polynomial degree for spine curve fitting
- `_DENSE_SAMPLES = 400` — sampling density for slope computation

## Conventions
- Input masks must already be remapped (0=bg, 1..17=T1..L5) — not raw dataset IDs
- Returns 0.0 when fewer than 4 vertebrae detected (tangent) or 5 (polynomial)
- Three methods: polynomial (original), tangent (recommended), endplate (PCA-based)
