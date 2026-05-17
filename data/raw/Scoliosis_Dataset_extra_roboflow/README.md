# Scoliosis Dataset — Extra (Roboflow scoliosis2)

A NEW dataset of 18 hand-labeled radiographs from the Roboflow Universe
[`scoliosis2`](https://universe.roboflow.com/college-wtsl5/scoliosis2-dvnfp)
dataset. Labels produced by collaborator (Jorge Oñate / team) via the same
GIMP / strict-palette workflow used for the v2 corrections — see
[`scripts/mask_correction_x2/mask_correction_workflow.py`](../../../scripts/mask_correction_x2/mask_correction_workflow.py)
and the wiki page [[Mask_correction_workflow]].

## Contents

| Folder | Files | Description |
|---|---|---|
| `Normal/` | 6 .jpg | N_72 .. N_77 (numbering continues v2's N_1..N_71) |
| `Scoliosis/` | 12 .jpg | S_207 .. S_217 + S_317 (continues v2's S_1..S_206) |
| `LabelMultiClass_ID_PNG/` | 18 .png | Single-channel ID masks (palette IDs 0..17 = bg + T1..L5) |
| `LabelMultiClass_Color_PNG/` | 18 .png | RGB-colored versions (intermediate GIMP outputs) |
| `indice_dataset.csv` | 1 | Per-case index with paths + vertebra completeness + status |

## Status

| status | n | note |
|---|---|---|
| ok | 15 | full T1..L5 labeled |
| warn | 3 | S_212, S_214, S_317 — all missing L5 (likely sacralization, same pattern as v2) |

No cases are currently `excluded`. The 3 `warn` cases can be reviewed
later via the same verification HTML workflow (see
`scripts/mask_correction_x2/verify_audit_cases.py`).

## Naming convention

- Radiographs: `<prefix>_<id>.jpg` (e.g. `N_72.jpg`, `S_207.jpg`)
- Masks: `LabelMulti_<prefix>_<id>.png` (e.g. `LabelMulti_N_72.png`)

Same as `Scoliosis_Dataset_v2_corrected/` so `mask_correction_workflow.py`
and the rest of the tooling Just Work.

## Palette

Strict T1..L5 17-color palette — identical to
[`scripts/mask_correction_x2/palette.json`](../../../scripts/mask_correction_x2/palette.json).
The mask validator only accepts IDs 0..17.

## How this compares to v2_corrected

| Aspect | v2_corrected | extra_roboflow |
|---|---|---|
| Source | MaIA Spanish-annotated dataset | Roboflow Universe `scoliosis2` |
| n cases | 250 | 18 |
| Image format | .jpg (variable res, ~241×878) | .jpg (mostly 2048×~600) — **higher resolution** |
| ID space | 1..17 (T1..L5) | 1..17 (T1..L5) — same |
| Cobb GT | yes (RadiographMetrics/) | **no** — Cobb metrics not provided |
| Curve CSV | yes | no |
| Patient ID space | N_1..N_71, S_1..S_206 | N_72..N_77, S_207..S_217+S_317 |

The two datasets are **disjoint** in patient_id space (no overlapping IDs)
so they compose cleanly into a merged trainable set.

## Combining with v2_corrected at training time

The trainer can read both datasets by concatenating their `indice_dataset.csv`:

```python
v2 = pd.read_csv('data/raw/Scoliosis_Dataset_v2_corrected/indice_dataset.csv')
ex = pd.read_csv('data/raw/Scoliosis_Dataset_extra_roboflow/indice_dataset.csv')
merged = pd.concat([v2, ex], ignore_index=True)
```

Note: v2's indice uses Spanish column names (`grupo`, `imagen`, `ruta_radiografia`...);
this dataset's indice uses English-equivalent paths (`image_path`,
`multiclass_mask_path`, `target_vertebrae_count`, `status`). A
small mapping adapter will be needed when first using both together.

## DVC tracking

Tracked via the top-level `data/raw.dvc` (just like v2). When the
directory hash refreshes, DVC catches the new files automatically.
