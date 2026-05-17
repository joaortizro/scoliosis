# Mask Correction x2 — DeepLabV3+ Pre-Mask + Manual GIMP Labeling

Goal: lift the 90 incomplete-GT cases from `data/processed/audit_v2_corrected/`
to fully-labeled T1..L5 (17-class) masks, producing
`data/raw/Scoliosis_Dataset_v2_corrected_x2/`.

Predicted impact (per [2026-05-11 audit](../../data/processed/audit_v2_corrected/)):
headline 5-fold Dice 0.6946 → ~0.78 once the 90 cases are fixed.

## Background

The original `Scoliosis_Dataset_v2_corrected/` was produced by Jorge's
**Phase 1** workflow: ID-mask → 17-color RGB → manual GIMP edit → strict
color validator → ID. That phase fixed 147 masks by **relabeling existing
spine pixels** that were fused under a single ID. It did NOT add new
vertebrae to masks that were missing them.

The 2026-05-11 audit found 90 cases where the v2_corrected masks still have
missing vertebrae (mid-spine gaps, missing L5). Those gaps need the
**Phase 2** workflow:

1. Run Jorge's DeepLabV3+ binary spine model (`best_model.pth`, Dice ~0.90)
   on the image → binary pre-mask of the spine region (no IDs).
2. Open the pre-mask + original radiograph in GIMP.
3. Color each visible vertebra T1..L5 using the strict palette (`palette.json`).
4. Run `mask_correction_workflow.py::convert_edited_color_to_id` → ID mask.
5. Output lands in `data/raw/Scoliosis_Dataset_v2_corrected_x2/`.

## Folder layout

```
scripts/mask_correction_x2/                    # tracked
├── README.md                                  # this file
├── palette.json                               # 17-color hex palette T1..L5
├── triage_audit_cases.py                      # NEW — classify the 90 cases
├── generate_premasks.py                       # NEW — DeepLabV3+ inference
└── mask_correction_workflow.py                # ADAPTED from Jorge
                                                # (RGB↔ID conversion + GIMP helpers)

ai/models/checkpoints/deeplabv3plus_binary/    # DVC-tracked (user-action: download)
└── best_model.pth                             # Jorge's binary spine model

data/processed/v2_corrected_x2_triage/         # DVC-tracked
└── audit_x2_triage.csv                        # one row per case, with fixability hint

data/processed/v2_corrected_x2_premasks/       # DVC-tracked
├── overlays/                                  # PNG overlays for visual inspection
├── masks_png/                                 # binary pre-masks (input to GIMP)
└── probability_maps/                          # DeepLabV3+ confidence maps

data/raw/Scoliosis_Dataset_v2_corrected_x2/    # DVC-tracked, final dataset
└── LabelMultiClass_ID_PNG/                    # only the 90 corrected files
                                                # (rest symlinks v2_corrected)

.local/mask_correction_x2/                     # gitignored, dev scratch
├── color_edit/                                # human-edited RGB intermediate
│   └── LabelMulti_S_NNN_COLOR_EDIT.png
└── logs/
    └── correction_log.csv                     # per-case audit trail
```

## Workflow per case

```bash
# 1. (one-time) generate pre-masks for all 90 triage cases
python scripts/mask_correction_x2/generate_premasks.py \
    --triage data/processed/v2_corrected_x2_triage/audit_x2_triage.csv \
    --model ai/models/checkpoints/deeplabv3plus_binary/best_model.pth \
    --out data/processed/v2_corrected_x2_premasks/

# 2. per-case manual loop (in GIMP, see below)
# 3. (per case) convert edited color → ID and validate
python -i scripts/mask_correction_x2/mask_correction_workflow.py
>>> convert_edited_color_to_id("LabelMulti_S_NNN")
>>> show_bad_colors_red("LabelMulti_S_NNN")  # if validator complains
```

## GIMP setup (one-time)

1. Install GIMP: <https://www.gimp.org/>
2. Load the project palette: `palette.json` (17 hex colors T1..L5)
3. Tool config:
   - Pencil: opacity 100, hardness 100, size 1 px
   - Eraser: opacity 100, hardness 100, hard-edge ON
   - Bucket Fill: threshold 0.0

**Hard rule: only use palette colors.** Any color outside the palette
fails strict validation. Antialiasing / partial opacity edges → fail.

## Color palette (T1..L5)

| ID | Vertebra | Hex      | ID | Vertebra | Hex      |
|----|----------|----------|----|----------|----------|
| 1  | T1       | #F2D10C  | 10 | T10      | #0CF268  |
| 2  | T2       | #EBF20C  | 11 | T11      | #0CF28F  |
| 3  | T3       | #C4F20C  | 12 | T12      | #0CF2B7  |
| 4  | T4       | #9CF20C  | 13 | L1       | #0CF2DE  |
| 5  | T5       | #75F20C  | 14 | L2       | #0CDEF2  |
| 6  | T6       | #4DF20C  | 15 | L3       | #0CB7F2  |
| 7  | T7       | #26F20C  | 16 | L4       | #0C8FF2  |
| 8  | T8       | #0CF219  | 17 | L5       | #0C68F2  |
| 9  | T9       | #0CF240  | 0  | bg       | (black)  |

## Triage assignment

To be filled in by `triage_audit_cases.py`. Per Jorge's expansion task split,
the 4-person rotation is Jorge / Fedys / Beto / Jonas. Same rotation can
apply here; ~90 cases → ~22 per person, ~5h each at 10 min/case.

## Dependencies on Jorge

| Artifact | Source | Status |
|---|---|---|
| `best_model.pth` (DeepLabV3+) | Google Drive `vertebra_mask_correction/` or `spine_pred/models/` | **TODO: download** |
| `mask_correction_workflow.py` (source) | Same Drive | **TODO: download + adapt PROJECT_ROOT paths** |
| Color palette source | Doc + Drive | Captured in `palette.json` (this folder) |

## References

- Charter: [[Propuesta_Proyecto_IBIO]] (in scoliosis-wiki)
- Audit: `data/processed/audit_v2_corrected/manual_correction_priority_top50.csv`
- Process doc: shared Google Doc (Jorge) — local copy at
  `/mnt/c/Users/ortiz/Downloads/Documentación del proceso de detección y corrección automática de estructuras vertebrales.md`
- Wiki: `[[2026-05-11_v2_gt_completeness_audit]]`, `[[Scoliosis_v2_corrected]]`
