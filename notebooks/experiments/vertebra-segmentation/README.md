# Vertebra Segmentation Experiments

This folder contains shared experiment notebooks for vertebra segmentation on AP scoliosis radiographs.

## Files
- `vertebra_yolov8_segmentation.ipynb` — main multiclass YOLOv8-seg experiment notebook
- `vertebra_binary_premask.ipynb` — binary premask experiment notebook
- `deeplabv3plus_binary_spine_premask.ipynb` - Binary premask experiment notebook, with advanced methods
- `monai_unet_binary_spine.ipynb` - Test for binary experiments (good results but not the best)

## Notes
- Paths must be configured inside each notebook before running.
- Corrected masks are prioritized over original masks when available.
- Large datasets, checkpoints, exported training runs, and other heavy artifacts are not stored in Git.