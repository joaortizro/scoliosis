# Path B — Detection-First Cobb Pipeline (Phase 3b + Q-22 chapter)

**Date**: 2026-05-10
**Branch target**: `feature/path-b-detection-cobb` (worktree)
**Spec author**: Claude (brainstorming session)
**Status**: draft pending user approval

## 1. Context & Background

The Phase 1.x segmentation-then-tangent pipeline plateaus at **Cobb MAE 8.16° ± 0.56°** (5-fold CV on v2-corrected, multitask v4 architecture) and **Dice 0.6946 ± 0.0205** (Phase 1.2 single-task). Two strong baselines now confirm this is a **data ceiling, not a model ceiling**:

- Phase 1.2 EncoderUNet (resnet34 + EMA + boundary λ=0.05 + ROI crop): 0.6946 ± 0.0205 across 5 folds.
- nnU-Net 2D auto-config (truncated 2026-05-10): fold 0 = 0.5774, fold 1 = 0.6908. Same Dice band as Phase 1.2.

Literature SOTA on Cobb MAE is **2.4–4.2°** (SpineNET 2026 = 1.44°, VCRLD-Net = 2.38°, YOLOv8-CBAM = 2.56°, Mazurowski 2025 = 4.17°, Seg4Reg+ 2022 = 4.2°). Every literature SOTA Cobb pipeline is **detection-first with corner-keypoint regression**, not segmentation-first with curve-tangent.

In addition, the existing geometric Cobb pipeline (`ai/evaluation/cobb.py::cobb_from_segmentation_tangent`) was found broken on this dataset — 16–31° MAE on GT masks alone (commit `cdcddf6`). The polynomial-fit + tangent geometry inherits a structural floor that does not improve with better masks.

This spec describes the pivot to a detection-first pipeline using:

- **Roboflow scoliosis2.v16i** (1,535 train + 100 valid + 101 test images, YOLO bbox format, vertebra-agnostic, CC BY 4.0) as pretraining data.
- **v2-corrected** (179 train + 45 val per fold across 5 folds, plus 25 sealed test) as fine-tune + evaluation data, leveraging the existing `ai/preprocessing/keypoints.py::multiclass_mask_to_keypoints()` (17 vertebrae × 4 corners = 68 keypoints per case).

## 2. Goals & Non-Goals

### Goals

- G1. Pivot Cobb computation from segmentation-then-tangent to **detection-first keypoint regression with endplate-slope Cobb**.
- G2. Achieve 5-fold mean Cobb MAE ≤ 5° on v2 (`metricas_cobb_resumen_recalculado.csv::angulo_cobb_deg`) — the thesis defense bar.
- G3. Produce a thesis methodology chapter (Q-22) on partial-coverage robustness, comparing segmentation-tangent vs detection-first degradation under cropped/partial X-ray conditions.
- G4. Document the empirical data-ceiling finding (two strong baselines plateau on 224 cases) as a methodology byproduct.

### Non-Goals (YAGNI)

- N1. No per-vertebra ID classification (T1 vs T2). Vertebra-agnostic detection is sufficient for endplate-slope Cobb; per-vertebra IDs are deferred unless 5-fold MAE > 5° and ID-aware ablation is needed.
- N2. No custom YOLO architecture. Use stock YOLOv8-Pose via `ultralytics` (`pip install ultralytics`).
- N3. No detection-head integration into the project's existing trainer (`ai/training/trainer.py`). Use ultralytics' standalone training loop. Keeps EncoderUNet lifecycle decoupled.
- N4. No retraining segmentation alongside detection. Orthogonal track.
- N5. No real-time inference optimization. Research-track only; eval throughput is sufficient.
- N6. No expansion to AASCE19 in this spec. AASCE19 access is still in the Q-04 escalation ladder.

## 3. Success Criteria

| Tier | Bar | Action on hit |
|---|---|---|
| **Sanity gate** | endplate-slope Cobb on v2 GT corners ≤ **3° MAE** vs `angulo_cobb_deg` (n ≈ 82 — scoliosis cases in the trainable set, the subset with both GT corners and Cobb GT) | Continue to training |
| **Floor (publish)** | 5-fold mean Cobb MAE < **8.16°** (beats v4-multitask) | Publishable as detection-first attempt + Q-22 chapter |
| **Thesis bar** | 5-fold mean Cobb MAE ≤ **5°** | Single-touch sealed-test eval on 25 holdout cases for thesis number |
| **SOTA-aspirational** | 5-fold mean Cobb MAE ≤ **4°** | Bonus — places work in the literature SOTA band |

5-fold std must be ≤ 1.5° MAE for results to be publishable.

Q-22 deliverable: stratified Cobb MAE table (vertebra-count buckets × severity buckets) showing detection-first degradation curve flatter than segmentation-tangent curve.

## 4. Architecture & Pipeline

```
v2 X-ray (any resolution)
   ↓
[1] preprocess: resize 512×256 (BILINEAR), normalize /255
   ↓
[2] YOLOv8-Pose detector (ultralytics) → list[(cx, cy, w, h, conf, kpt1..4_xy_visibility)]
       config: kpt_shape=[4, 3], freeze_backbone=True (during v2 fine-tune)
   ↓
[3] postprocess:
       a. confidence filter τ=0.25
       b. top-N=20 cap (anatomically plausible upper bound)
       c. PCA-axis projection ordering (fallback to y-sort if PCA variance ratio < 0.95)
   ↓
[4] endplate-slope Cobb (NEW — replaces polyfit + tangent):
       a. per vertebra v in ordered chain:
            upper_slope[v] = atan2(kpt_TR.y - kpt_TL.y, kpt_TR.x - kpt_TL.x)   # in radians, sign convention: image-y axis flipped
            lower_slope[v] = atan2(kpt_BR.y - kpt_BL.y, kpt_BR.x - kpt_BL.x)
            mean_slope[v]  = (upper_slope[v] + lower_slope[v]) / 2
       b. Cobb angle = max(mean_slope) - min(mean_slope) over the ordered chain  (degrees)
       c. apex vertebra = argmax_v |d²(mean_slope)/dv²|  (vertebra at max second-derivative of slope sequence)
       d. upper inflection vertebra = argmax_v above the apex (steepest slope above)
            lower inflection vertebra = argmin_v below the apex (steepest slope below)
   ↓
output: cobb_angle_deg, apex_vertebra_idx, all_endplate_slopes
```

### Module layout

```
ai/
├── detection/                          # NEW
│   ├── __init__.py                     # public exports
│   ├── yolo_adapter.py                 # ultralytics wrapper (load, predict)
│   ├── train_yolo.py                   # training driver
│   ├── data_conversion.py              # v2 multiclass mask → YOLO-Pose label; Roboflow bbox → YOLO-Pose dummy-kpt label
│   └── postprocess.py                  # confidence filter, top-N, PCA-axis ordering
├── evaluation/
│   ├── cobb.py                         # existing — leave as-is for back-compat
│   └── cobb_endplate.py                # NEW — endplate-slope Cobb implementation
└── __init__.py                         # add re-exports: detect_vertebrae, cobb_from_detections

scripts/
├── train_yolo_roboflow.py              # Phase 3b.2 driver
├── train_yolo_v2_finetune.py           # Phase 3b.3, 3b.4 driver (per-fold)
├── eval_cobb_phase3b.py                # 5-fold Cobb MAE eval + Q-22 stratified table
└── sanity_endplate_cobb_on_gt.py       # Phase 3b.1 sanity gate

tests/
├── test_endplate_cobb_on_gt.py         # NEW — sanity gate, hard CI gate ≤ 3° MAE
├── test_yolo_pose_label_conversion.py  # NEW — round-trip check
├── test_roboflow_filter.py             # NEW — coverage filter correctness
├── test_no_leakage.py                  # existing — extend with sealed-test asserts for Roboflow + v2 fine-tune sets
└── test_architecture.py                # existing — extend with `ai/detection/` boundary checks
```

### Hexagonal-architecture compliance

- `ai/detection/` is part of the pure ML library (no FastAPI, no SQLAlchemy, no server imports).
- The `ultralytics` dependency is confined to `ai/detection/yolo_adapter.py`. All other modules import from `yolo_adapter` interfaces, not `ultralytics` directly.
- Server's `infrastructure/adapters/secondary/ml/` is not modified in this spec — Path B is research-track until thesis numbers land.
- New public API (`ai/__init__.py`):
  - `detect_vertebrae(image: np.ndarray) -> list[Detection]`
  - `cobb_from_detections(detections: list[Detection]) -> CobbResult`

## 5. Data Plan

### 5.1 Roboflow ingestion (one-time)

- Source: existing download at `C:\Users\ortiz\Downloads\archive (1)\scoliosis2.v16i.tensorflow` (250 MB, 1,736 images).
- Move to `data/raw/roboflow_scoliosis_v16/`.
- `dvc add data/raw/roboflow_scoliosis_v16/` + `dvc push` (~250 MB to S3).
- **Coverage filter**: include only images with ≥ 14 detected vertebrae per the Roboflow label files. Drops the partial/cropped Roboflow cases from the pretrain set; preserves them for Q-22 cross-eval if useful later. Filter applied at training-time via `data/processed/roboflow_filtered_train.txt` (DVC-tracked).
- **No filename overlap with v2** verified during 2026-05-10 audit ([[2026-05-10_roboflow_scoliosis_audit]]).

### 5.2 v2 → YOLO-Pose label conversion

New module `ai/detection/data_conversion.py`:

```python
def multiclass_mask_to_yolo_pose(mask: np.ndarray) -> list[YoloPoseLabel]:
    """For each nonzero class (1..17 = T1..L5):
       - bbox = tight axis-aligned box around class pixels (cx, cy, w, h, normalized)
       - keypoints = 4 corners from multiclass_mask_to_keypoints(mask)[class_idx]
       - visibility = 0 if NaN else 2
       Returns list of labels, one per vertebra present in the mask.
       class_id = 0 (vertebra-agnostic) for all entries.
    """

def roboflow_to_yolo_pose(bbox_label_path: Path) -> list[YoloPoseLabel]:
    """Re-format Roboflow bbox labels with 4 dummy keypoints at bbox corners,
       visibility = 0. Kpt loss is masked during pretrain (visibility=0 ⇒ ignored).
    """
```

- Run once per fold split during preprocessing: `data/processed/yolo_pose_labels/fold_{0..4}/{train,val}/labels/*.txt` — DVC-tracked.
- Roboflow conversion produces `data/processed/yolo_pose_labels/roboflow_pretrain/labels/*.txt` — DVC-tracked.

### 5.3 Splits

- **5-fold CV on v2**: reuse existing `ai.training.splits.make_cv_folds(clean_index_csv, n_folds=5, seed=42)`. Per-fold split = 179 train + 45 val.
- **Sealed test holdout**: `data/processed/audit_v2_corrected/test_holdout.csv` (25 cases). Single-touch after 5-fold gate clears 5° MAE.
- **Roboflow split**: use Roboflow's pre-defined train/valid/test splits (1,535 / 100 / 101). Pretrain on train, validate during pretrain on Roboflow's valid.

### 5.4 Cobb GT

- Source: `data/raw/Scoliosis_Dataset_v2_corrected/RadiographMetrics/metricas_cobb_resumen_recalculado.csv::angulo_cobb_deg`.
- Available for all 179 scoliosis cases (normal cases have no ground-truth Cobb — excluded from Cobb MAE computation; included in detection mAP).

## 6. Q-22 Methodology Chapter — Partial-Coverage Robustness

### 6.1 Hypothesis

Detection-first pipelines degrade more gracefully than segmentation-tangent pipelines under partial spinal coverage (cropped X-rays showing only part of T1..L5). The detection pipeline emits whatever vertebra centroids are visible and computes Cobb on a variable-length list; the segmentation-tangent pipeline relies on a curve fit that becomes unstable when only mid-spine is shown.

### 6.2 Methodology

Eval-time crop synthesis. No additional training data needed.

For each v2 test case (across all 5 folds, plus the sealed test holdout if reached), generate four cropped variants:

| Bucket | Vertebrae visible | Crop region |
|---|---|---|
| `full` | T1..L5 (17 visible) | uncropped reference |
| `top_15` | T1..T15 (15 visible) | top 88% of image height |
| `mid_10` | T8..L1 (10 visible) | middle 59% |
| `bot_8` | T12..L5 (8 visible) | bottom 47% |

Crop is applied **post-inference** by filtering the predicted keypoint set to the visible-bucket subset. This isolates the geometry-layer's robustness from the detector's robustness.

For each (case, bucket) pair, compute Cobb MAE using:
- (A) Phase 1.2 segmentation-tangent pipeline
- (B) Phase 3b detection-first endplate-slope pipeline

Stratify by SOSORT severity: mild (< 25°), moderate (25–40°), severe (> 40°).

### 6.3 Output

- `experiments/results/phase3b_q22_stratified.csv` — long-form: `case_id, bucket, severity, pipeline, cobb_pred_deg, cobb_gt_deg, error_deg`.
- `experiments/results/phase3b_q22_summary.csv` — pivot table: `pipeline × bucket × severity → MAE, MdAE, n`.
- Figure (matplotlib): degradation curves overlaid (x-axis = vertebra count visible, y-axis = MAE, two lines for the two pipelines).

## 7. Testing Strategy

| Test | What it checks | Severity |
|---|---|---|
| `test_endplate_cobb_on_gt.py` | endplate-slope Cobb on v2 GT corners ≤ 3° MAE vs `angulo_cobb_deg` (n=152) | **HARD CI GATE** |
| `test_yolo_pose_label_conversion.py` | Round-trip: keypoints → label → reload → match within 1px | unit |
| `test_roboflow_filter.py` | Filtered Roboflow set: all images have ≥ 14 vertebrae; no overlap with v2 filenames | unit |
| `test_no_leakage.py` (extended) | Sealed 25-case test never appears in Roboflow pretrain or v2 fine-tune sets across all 5 folds | hard CI |
| `test_architecture.py` (extended) | `ai/detection/` does not import from `server/`, `fastapi`, `sqlalchemy`. `ultralytics` confined to `ai/detection/yolo_adapter.py`. | architectural fitness |
| `test_pca_ordering.py` | PCA-axis ordering preserves head-to-tail order on synthetic ground-truth chains; falls back to y-sort when configured threshold is hit | unit |
| `test_cobb_endplate_invariance.py` | Endplate-slope Cobb is invariant under image scale and per-axis flip (with sign convention) | unit |

All tests run via `pytest tests/ -v`. The `test_endplate_cobb_on_gt.py` sanity gate must pass before training. If it fails, the geometry layer is broken and training will inherit the floor — fix the geometry first.

## 8. Phase Plan & Cost

| Phase | Description | Wall | Cost | Sentinel | Blocking gate |
|---|---|---|---|---|---|
| 3b.0 | Implement endplate-slope Cobb in `ai/evaluation/cobb_endplate.py` | 4 h dev | $0 | code merged | — |
| 3b.1 | Sanity gate: endplate Cobb on v2 GT corners | 2 h | $0 | `experiments/results/sanity_endplate_cobb_gt.json` | **MAE ≤ 3°** else stop |
| 3b.2 | Pretrain YOLOv8-Pose on Roboflow (bbox-only loss) | ~1 h on g6e | ~$2 | `experiments/results/yolo_roboflow_pretrain.json` | mAP@0.5 ≥ 0.85 on Roboflow valid |
| 3b.3 | Single-fold ablation: v2-only vs v2 + Roboflow-pretrain on fold 0 | 2 h | ~$3 | `experiments/results/phase3b_fold0_ablation.json` | **Δ MAE ≥ 0.5°** else drop Roboflow |
| 3b.4 | 5-fold fine-tune with chosen pretrain strategy (freeze backbone, mosaic + HSV + ±15° rotation) | ~6 h on g6e | ~$11 | `experiments/results/phase3b_5fold.json` | 5-fold mean MAE ≤ 5° |
| 3b.5 | Q-22 partial-coverage stratified eval (synth crops at eval time) | 2 h | $0 | `experiments/results/phase3b_q22_stratified.csv` + `_summary.csv` | — |
| 3b.6 | If 5-fold MAE ≤ 5°: single sealed-test eval on 25 holdout cases | 30 min | ~$1 | `experiments/results/phase3b_sealed_test.json` | — |

**Total**: ~17 h wall, ~$17 USD on g6e ($1.86/hr).

Compared to the original architecture-page estimate ($10–12), the +$5–7 buys two kill-switches (3b.1 sanity gate + 3b.3 ablation) that prevent burning $30+ on a broken geometry layer or unhelpful pretraining.

## 9. Risks & Mitigations

Ranked by impact (advisor critique 2026-05-10).

| Rank | Risk | Mitigation |
|---|---|---|
| 1 | Endplate-slope Cobb produces 10°+ MAE floor regardless of detector quality (geometry layer broken) | Phase 3b.1 sanity gate. Hard CI test. If MAE > 3° on GT corners, stop and fix geometry before any training. |
| 2 | Detector overfits on 224 v2 cases even with Roboflow pretraining | Freeze YOLO backbone during v2 fine-tune via ultralytics' `freeze` parameter: freeze layers 0..9 (the CSPDarknet backbone), unfreeze the neck + head. Augmentation: keep ultralytics' defaults (`mosaic=1.0`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=15`, `scale=0.5`, `flipud=0.0`, `fliplr=0.5`). Report 5-fold std; reject result if std > 1.5°. |
| 3 | Roboflow domain gap (Turkish clinic vs MaIA Spanish v2) hurts pretrain transfer | Phase 3b.3 single-fold ablation. If Δ MAE < 0.5° between v2-only and v2 + Roboflow-pretrain, drop Roboflow and skip 3b.4's pretrained variant. |
| 4 | Roboflow has natural partial coverage (5–13 vertebrae cases) that pollute pretrain | Coverage filter ≥ 14 vertebrae applied via `data/processed/roboflow_filtered_train.txt`. Reserved partial-coverage Roboflow cases available for Q-22 cross-eval as stretch goal. |
| 5 | YOLOv8-Pose requires keypoint visibility flags — Roboflow has no keypoints, only bboxes | Use dummy keypoints at bbox corners with visibility=0 during pretrain. ultralytics skips kpt loss for visibility=0 keypoints. Verify with a unit test. |
| 6 | PCA-axis ordering fails on severe S-curves where principal axis is ambiguous | Variance-ratio threshold 0.95 falls back to y-sort. Tested in `test_pca_ordering.py`. If failure rate > 5% of cases, log a warning and consider arc-length-along-curve ordering as v2 of the design. |
| 7 | nnU-Net checkpoint disk usage from prior session left ~1 GB unused on local | Leave for now. Path B uses a separate checkpoint dir under `ai/models/checkpoints/yolo_vertebra/`. nnU-Net artifacts are DVC-tracked and can be pruned later. |

## 10. Reproducibility & Provenance

- Each run produces `cfg.json` capturing: backbone (`yolov8n-pose`), `kpt_shape: [4, 3]`, image size (512×256), batch, epochs, lr, freeze_backbone (bool), data sources (`roboflow_v16_filtered_vge14` / `v2_5fold_split_<hash>`), pretrain checkpoint hash.
- `dvc add ai/models/checkpoints/yolo_vertebra/<TIMESTAMP>_<cfg_hash>/` per run + `dvc push` (per `feedback_save_model_weights` memory rule).
- MLflow logs metrics + params + artifacts.
- Sentinel JSON files committed to `experiments/results/` per existing convention.
- Wiki updates per `feedback_wiki_after_run`: append to `log.md`, update `experiments/_index.md` leaderboard, create `experiments/2026-05-10_phase3b_*.md` per run, update `hot.md` active_track.

## 11. Open Questions

| ID | Question | Resolution path |
|---|---|---|
| Q-PB-1 | Should YOLO backbone be frozen during v2 fine-tune (head-only) or unfrozen (full fine-tune)? | Default: frozen. Settle empirically with a single-fold ablation if 5-fold std > 1.5°. |
| Q-PB-2 | Image size: 512×256 matches Phase 1.x; ultralytics default is 640×640 — which gives better mAP / Cobb MAE? | Default: 512×256. If detection mAP@0.5 < 0.85 in 3b.2, retry at 640×640. |
| Q-PB-3 | PCA-axis ordering threshold (variance ratio < 0.95 → fallback to y-sort) — is 0.95 the right number? | Tune on v2 5-fold: choose threshold that maximizes ordering correctness on GT (n=152 with ground-truth corner ordering). |
| Q-PB-4 | Q-22 partial-coverage: should crops be deterministic (3 fixed buckets) or random (random vertebra-window per case)? | Default: deterministic (4 buckets defined in §6.2). Random crop is a sensitivity ablation, not the headline number. |
| Q-PB-5 | If v2 + Roboflow-pretrain underperforms v2-only (3b.3 Δ < 0): is that a finding or a bug? | Treat as finding — document as evidence that within-domain data > cross-domain pretrain at this scale. Still report. |

## 12. Cross-References

- [[VertebraDetectorCentroidCobb]] — original architecture page (status: planned). Update to `status: active` and revise to reflect endplate-slope Cobb after this spec lands.
- [[2026-05-10_roboflow_scoliosis_audit]] — Roboflow facts and quality flags.
- [[2026-05-10_nnunet_2d_5fold_truncated]] — nnU-Net negative result that motivated the pivot.
- [[2026-05-08_phase0_ec2_rerun]] — Phase 1.2 reference numbers.
- [[Open_Questions]] §Q-22 — partial-coverage thesis chapter.
- [[Open_TODOs]] §ADR-test-set-access-policy — sealed-test gating policy (still open).

---

**Spec status**: draft. Awaiting user review per brainstorming-skill review gate. After approval, implementation plan is generated via `superpowers:writing-plans` and execution begins on the `feature/path-b-detection-cobb` worktree.
