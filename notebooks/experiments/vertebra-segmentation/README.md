# Vertebra Segmentation Experiments

This folder contains shared experiment notebooks for vertebra segmentation on AP scoliosis radiographs.

## Files
- `vertebra_yolov8_segmentation.ipynb` — main multiclass YOLOv8-seg experiment notebook
- `vertebra_binary_premask.ipynb` — binary premask experiment notebook

## Notes
- Paths must be configured inside each notebook before running.
- Corrected masks are prioritized over original masks when available.
- Large datasets, checkpoints, exported training runs, and other heavy artifacts are not stored in Git.