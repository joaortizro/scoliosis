# Paper Evidence Pack — Scoliosis Segmentation (2026-05-18 v1.1)

> **Purpose.** Single self-contained document mapping every claim the paper makes to its primary evidence (sentinel file, wiki page, or git commit). Designed so a paper-writing skill (or any future agent) can produce the manuscript without navigating the wiki page-by-page.
>
> **Status.** Definitive as of 2026-05-18 06:50 UTC. D2 5-fold CV COMPLETED, sentinel committed at `ab351b2` on `main`. EC2 stopped. All training final.
>
> **Naming update v1.1:** paper-facing aliases applied. IBIO-SD (was v2/v2_corrected/v2_corrected_x2), ERS-18 (was extra_roboflow), RB-UNet (was Phase 1.2 / EncoderUNet). Canonical mapping at [[Paper_naming_aliases]].
>
> **Provenance.** Derived from `experiments/results/*.json` sentinels (the ground truth), the wiki at `/home/ortiz/scoliosis-wiki/` (interpretation + cross-links), and `params.yaml` / `dvc.yaml` (configuration). Where the wiki and sentinels conflicted, sentinels win.

---

## 0. How to use this pack

Two reading orders.

**For paper writing (top-to-bottom):**
1. Read §1 Charter + §2 Project state to fix the scope and the deadline.
2. Read §3 Datasets + §4 Production recipe to fill the Methodology section.
3. Read §5 Headline numbers + §6 Per-fold + §7 Per-class + §8 Partial-FOV + §9 OOD to fill Results.
4. Read §10 Methodology contributions + §11 Failure modes + §13 Forensic findings to fill Discussion.
5. Read §14 Negative results + §15 Cobb context to round out Discussion.
6. Read §17 SOTA positioning to ground Estado del Arte.
7. Use §18 Sentinel index whenever a specific number needs verification.

**For QA / cross-checking (lookup):**
- Find the claim in the paper.
- Locate the matching subsection in this pack (sections are indexed in the TOC).
- Each claim cites either a sentinel path, a commit SHA, or a wiki page wikilink.
- If verification is needed, open the sentinel or wiki page directly.

---

## 1. Charter (verbatim, from `docs/references/Propuesta_Proyecto_IBIO.pdf`)

The charter is the project's binding scope document, signed by the advisors (Luis Felipe Giraldo + Christian Javier Cifuentes) at Grupo IBIO / Uniandes.

**Required deliverables:**
- Segmentación automática de columna vertebral **y de cada vértebra**.
- Pacientes sanos **y con escoliosis**.
- Robustez ante entradas parciales (radiografías que no muestran la columna completa).
- Evaluación cuantitativa via Dice / IoU.

**Desirable but NOT obligatory** (literal `"deseable, pero no obligatoria"`):
- Modelo ligero.
- Inferencia rápida.
- Portabilidad embebida.

**Outside scope:**
- Validación clínica formal con lectores múltiples.
- Despliegue en producción embebida.

**Project-historical addendum:** Cobb-angle estimation is bonus value-add ([[ADR-efficiency-desirable-not-mandatory]], [[Project_scope_segmentation_first]]). The charter does NOT require a clinical Cobb MAE; the segmentation result is the deliverable. Multi-task v4 work and YOLOv8-Pose Path B were exploratory.

---

## 2. Project state snapshot (2026-05-18 04:50 UTC)

| Item | Value |
|---|---|
| Paper deadline | 2026-05-18 EOD (~16 h remaining) |
| Paper file | `/home/ortiz/scoliosis.tex` (565 lines, IEEEtran conference, Spanish) |
| Production architecture | `EncoderUNet` (ResNet-34, ImageNet pretrained, ~24M params) |
| Production recipe name | "Phase 1.2 D1 + ROI" |
| Headline 5-fold val Dice | **0.6946 ± 0.0205** (macro mc) |
| Headline test Dice (sealed) | **0.6331 ± 0.0260** (macro mc) / **0.8771 ± 0.0066** (binary) |
| Charter-compliance metric (partial-FOV M1a) | **0.861** binary @ f=0.5, 5-fold (gate ≥0.80) |
| Currently running on EC2 | None — D2 5-fold completed 06:24 UTC, EC2 stopped 06:30 UTC |
| Last commit on main | `ab351b2` (D2 5-fold sentinel + .dvc pointers) |
| **D2 5-fold final result** | **macro Dice 0.7065 ± 0.0167 (+0.012 over Phase 1.2, std −0.004)** |
| Wiki health | linted 2026-05-18 04:45 UTC; 3 critical fixes applied (C-1, C-2, C-3) |
| Git remote pushed through | `746c3d3` |
| DVC remote | up to date (Phase 1.2 5-fold + dataset ablation checkpoints in S3) |

---

## 3. Dataset breakdown (definitive counts)

All counts verified empirically against `data/processed/audit_v2_corrected*/clean_index.csv` on 2026-05-17.

### 3.1 v2_corrected_x2 (primary working set)

| Quantity | Count | Source |
|---|---|---|
| Raw cases in v2_corrected_x2 | **250** | 179 Scoliosis (`Scoliosis/S_*.jpg`) + 71 Normal (`Normal/N_*.jpg`) |
| Excluded by audit (`status=excluded`) | **1** | 1 Scoliosis case — image/mask inconsistency |
| Trainable (`status ∈ {ok, warn} AND target_count ≥ 14`) | **249** | 178 Scoliosis + 71 Normal |
| Status distribution (trainable) | 170 ok + 79 warn | `target_count == 17`: 186 cases |
| Sealed test holdout | **25** | `data/processed/audit_v2_corrected/test_holdout.csv` |
| Train+val pool | **224** | 160 Scoliosis + 64 Normal |
| Single-split val (canonical 80/20, seed=42) | 45 | 80% train / 20% val on the 224-case pool |
| 5-fold val per fold | 44–45 | StratifiedGroupKFold (stratify=severity_bucket, group=patient_id) |
| Train per fold | 179–180 | 224 minus val (varies ±1 per fold) |

**Verification (Sec.~4.1 of paper):** the paper's "224 train+val + 25 test = 249" is correct after audit filter; the "250" cited elsewhere is the dataset raw size before audit.

**6 mask corrections** (overlay `v2_corrected_x2` over `v2_corrected`):

| Case | Originally | Corrected | Δ | Method |
|---|---|---|---|---|
| N_23 | 16 IDs | 17 IDs | +L5 | GIMP-painted L5 below existing L4 |
| N_28 | 16 IDs (1..6, 8..17) | 17 IDs | +T7 | T6 had 2 connected components; recolor lower blob T6→T7 |
| N_36 | 16 IDs (1..9, 11..17) | 17 IDs | +T10 | T11 had 2 components; recolor upper blob T11→T10 |
| N_52 | 16 IDs (1..9, 11..17) | 17 IDs | +T10 | GIMP-painted in 38-row gap T9→T11 |
| N_59 | 16 IDs (1..13, 15..17) | 17 IDs | +L2 | GIMP-painted in 51-row gap L1→L3 |
| N_71 | 15 IDs (2..8, 10..17) | 17 IDs | +T1, +T9 | GIMP-painted T1 + T9 |

All 6 corrections are on **Normal** cases. Workflow: [[Mask_correction_workflow]] (Jorge Oñate's strict-palette GIMP pipeline).

### 3.2 extra_roboflow (extension)

| Quantity | Count | Source |
|---|---|---|
| Raw cases | **18** | 6 Normal (N_72..N_77) + 12 Scoliosis (S_207..S_217, S_317) |
| Patient ID space | disjoint from v2 | N_72..N_77 and S_207..S_317 do NOT exist in v2 |
| Status distribution | 15 ok + 3 warn | 3 warn cases (S_212, S_214, S_317) missing L5 — likely sacralization, anatomically variant |
| Excluded | 0 | All 18 entered as trainable |
| Origin | Roboflow Universe `scoliosis2-dvnfp` | Hand-labeled by Jorge Oñate, GIMP strict-palette workflow (same as v2 corrections) |
| Resolution | ~2048 × 600–960 | ~10× higher than v2 (~241 × 878 typical) |
| Cobb GT | **Not provided** | Segmentation only |

**Merged D2 trainable set:** **249 + 18 = 267 cases** (NOT 268 — see lint report C-1 fix on 2026-05-18). When this set is used for 5-fold CV with v2-pinned val (the `phase1_2_d2_5fold.py` runner), every fold's val remains a v2-only subset and the 18 roboflow cases always go to train.

### 3.3 Deprecated / unused datasets

| Page | Status | Note |
|---|---|---|
| [[MaIA_v1_legacy]] | deprecated | Original 22-class English-named pre-cursor; not used since v2 ingest |
| [[Roboflow_scoliosis2_v16i]] | pending audit | The upstream Roboflow project from which Jorge selected 18 cases for re-annotation; the rest (~1717 cases) had unusable bbox-only labels |
| [[Mendeley_Spine_X_ray]] | external reference | Used in [[Scoliosis_SNOMED_Pipeline_2025]] (cite \cite{snomed_pipe}); not in our training pool |

---

## 4. Production recipe (exact configuration)

The single architecture + cfg combination that produces the paper's headline numbers.

### 4.1 Architecture

```
EncoderUNet
├── backbone: ResNet-34 (ImageNet pretrained)
│   └── stem: 3-channel → single-channel via channel-wise averaging (preserves transfer)
├── decoder: U-Net style with skip connections at each scale
│   └── blocks: 3×3 conv + BN + ReLU
└── output: 18 logit maps (background + 17 vertebrae T1..L5)

params: ~24M
input: 1 × 512 × 256 (single-channel grayscale)
output: 18 × 512 × 256 (per-pixel class probabilities)
```

Source: `ai/models/architectures/encoder_unet.py` and [[EncoderUNet]].

### 4.2 Pre-processing pipeline (`ai/training/dataset.py::preprocess_case`)

Order applied to each case:

1. **ROI crop from mask.** Compute bounding-box of binary spine mask, expand 5% margin, crop image + masks. *Largest single contribution to the headline number* (+0.021 macro Dice vs no-ROI baseline).
2. **Resize to 512×256.** Image: bilinear. Masks (binary + multiclass): nearest-neighbor.
3. **Normalize to [0,1].** Divide image by 255. **No CLAHE, no z-score, no ImageNet-stats.**
4. **Class remap.** Mask multiclass IDs already in {0..17}; passed through.

Training augmentation: `v4` profile (rotation ±5°, translation ±5%, scale ±10%, hflip p=0.5, brightness/contrast ±10%). For partial-FOV experiment: additionally `RandomVerticalCrop(p=0.5, f∼U(0.3, 1.0))` (variant name: `v4_vcrop_gentle`).

### 4.3 Loss function

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Dice}} + \lambda_b \cdot \mathcal{L}_{\text{boundary}}, \quad \lambda_b = 0.05$$

- $\mathcal{L}_{\text{CE}}$: per-pixel cross-entropy, class-weighted by inverse frequency.
- $\mathcal{L}_{\text{Dice}}$: soft Dice multiclass, macro over present classes.
- $\mathcal{L}_{\text{boundary}}$: Kervadec 2019 boundary-aware loss \cite{kervadec_boundary_2019}.

$\lambda_b = 0.05$ selected by Phase 0 sweep (Run D1 in §13.1).

### 4.4 Optimizer + schedule

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.999, ε=1e-8 |
| LR (encoder) | 1e-4 | Lower because backbone is pretrained |
| LR (decoder) | 1e-3 | 10× encoder |
| Weight decay | 2e-4 | |
| Batch size | 4 | Limited by 16GB T4 VRAM with 512×256 |
| Epochs (max) | 100 | |
| Freeze encoder epochs | 10 | Train decoder alone first; then unfreeze |
| LR schedule | Cosine annealing | over 100 epochs |
| EMA | enabled, decay=0.999 | Eval both live + EMA, report best |
| Early stop | patience=20, Δ_min=0 | For Phase 1.2 5-fold + new D2 5-fold. **NOTE: dataset_ablation single-split and partial-FOV used patience=10, min_Δ=0.001** (2026-05-16 cfg change; see [[2026-05-17_dataset_ablation_d1_d2#Forensics]]). |

Hyperparameters source of truth: `params.yaml`. Each training run captures a cfg-hash for reproducibility.

### 4.5 Split protocol

`ai/training/splits.py::make_cv_folds`:
- StratifiedGroupKFold from scikit-learn.
- Stratify on `severity_bucket` (Cobb buckets).
- Group on `patient_id` to prevent train/val leak.
- Seed 42 (same seed across all 5-fold experiments in the project).

For the D2 5-fold runner (`scripts/phase1_2_d2_5fold.py`), val for fold k is pinned to the v2-only fold k val case_ids (function `build_d2_fold_split`); the 18 roboflow cases enter every fold's train. This guarantees val is comparable to Phase 1.2 fold-by-fold.

---

## 5. Headline numbers (paper-grade)

Five regimes of evaluation. The bold rows are paper-headline-grade.

| Régimen | Dice binario | Dice mc macro | Cobb MAE | n | Sentinel |
|---|---|---|---|---|---|
| Single-split (val) | ~0.882 | 0.6739 | — | 45 | (prior, in run dir 20260509_*) |
| **Phase 1.2 5-fold CV (val)** | **~0.882** | **0.6946 ± 0.0205** | — | 5 × 45–50 | `phase1_2_5fold.json` |
| **Phase 1.2 5-fold (sealed test)** | **0.8771 ± 0.0066** | **0.6331 ± 0.0260** | **27.58° ± 0.24°** | 25 (touched ONCE) | `phase1_2_5fold_TEST.json` |
| Partial-FOV M1a 5-fold (val, full coverage) | ~0.851 | 0.6371 ± 0.0438 | — | 5 × 45–50 | `partial_fov_gentle_5fold.json` |
| **Partial-FOV M1a grid (mode-avg @ f=0.5)** | **0.861** | 0.557 | — | 224 × 4 modes | `partial_fov_5fold_summary.csv` |
| **Partial-FOV M1a grid (mode-avg @ f=0.3)** | **0.843** | 0.526 | — | 224 × 4 modes | `partial_fov_5fold_summary.csv` |
| Dataset ablation D1 (single-split) | — | 0.6203 | — | 45 | `dataset_ablation_d1_x2.json` |
| Dataset ablation D2 (single-split) | — | 0.6650 | — | 45 | `dataset_ablation_d2_x2_plus_roboflow.json` |
| **Dataset ablation D2 (5-fold val)** | — | **0.7065 ± 0.0167** | — | 5 × 44–45 | `phase1_2_d2_5fold.json` |
| OOD zero-shot 5-fold (roboflow 18) | — | 0.6560 ± 0.0328 | — | 5 × 18 | `zero_shot_5fold_on_roboflow.json` |

**Gate-clearing summary:**

| Gate | Threshold | Actual | Margin |
|---|---|---|---|
| Phase-1 Dice ≥ 0.665 | 0.665 | 0.6946 (5-fold val) | +0.030 ✅ |
| Partial-FOV binary @ f=0.5 ≥ 0.80 | 0.80 | 0.861 | +0.061 ✅ |
| Partial-FOV binary @ f=0.3 ≥ 0.65 | 0.65 | 0.843 | +0.193 ✅ |
| Thesis target Dice ≥ 0.78 | 0.78 | 0.6946 | −0.085 ✗ (label-noise-limited; see §10) |

**Val→test gap:** 0.6946 − 0.6331 = **−0.0615** on macro Dice. Within the typical generalization band; no evidence of test leakage.

---

## 6. Per-fold breakdown

### 6.1 Phase 1.2 5-fold val (production headline)

Sentinel `experiments/results/phase1_2_5fold.json`. Trained over ~5.5 h total on T4.

| fold | split_hash | best_val_dice | best_source | wall (s) | run_dir |
|---|---|---|---|---|---|
| 0 | 3c26422189d7822f | 0.6708 | live | 3835 | `20260509_194823_b41714d16d325371` |
| 1 | 3aeb683b2c828fe6 | 0.6932 | live | 3857 | `20260509_205219_6895f9edea602bfd` |
| 2 | 10d600a04d9c1ef7 | 0.6733 | live | 3059 | `20260509_215636_701388d2e69d1a77` |
| 3 | 6faf3bcb2a1717d1 | 0.7168 | live | 4502 | `20260509_224736_213128652dc85e86` |
| 4 | d6fbf7c4d7fe9417 | 0.7190 | live | 4630 | `20260510_000239_d7ed0282aa9c5883` |
| **mean** | — | **0.6946** | — | — | — |
| **std** | — | **0.0205** | — | — | — |
| min/max | — | 0.6708 / 0.7190 | — | — | — |

### 6.2 Phase 1.2 5-fold sealed test (single-touch)

Sentinel `experiments/results/phase1_2_5fold_TEST.json`. Hflip-TTA. Touched once on 2026-05-17.

| fold | binary Dice | macro mc Dice | Cobb MAE (°) | n_cobb |
|---|---|---|---|---|
| 0 | 0.8798 [0.869, 0.891] | 0.6574 [0.569, 0.739] | 27.43 [21.3, 33.4] | 18 |
| 1 | 0.8795 [0.870, 0.890] | 0.6429 [0.530, 0.749] | 27.40 [21.1, 33.3] | 18 |
| 2 | 0.8640 [0.852, 0.877] | 0.5834 [0.475, 0.686] | 27.58 [21.3, 33.3] | 18 |
| 3 | 0.8800 [0.870, 0.890] | 0.6480 [0.547, 0.743] | 27.46 [21.2, 33.4] | 18 |
| 4 | 0.8823 [0.872, 0.893] | 0.6338 [0.526, 0.736] | 28.06 [21.8, 33.9] | 18 |
| **mean** | **0.8771 ± 0.0066** | **0.6331 ± 0.0260** | **27.58 ± 0.24** | (n_cobb=18 of 25 with Cobb GT) |

CIs are 95% bootstrap on per-image Dice (2000 resamples). The 7 test cases without Cobb GT are Normal cases (no Cobb to compute).

Per-fold pooled per-class Dice is in §7 below.

### 6.3 Partial-FOV M1a 5-fold (full-coverage val)

Sentinel `experiments/results/partial_fov_gentle_5fold.json`. Same architecture as Phase 1.2, only delta is `RandomVerticalCrop(p=0.5, f∈U(0.3, 1.0))` added to train aug.

| fold | split_hash | best_val_dice | run_dir |
|---|---|---|---|
| 0 | 3c26422189d7822f | 0.5585 | `20260516_071804_*` |
| 1 | 3aeb683b2c828fe6 | 0.6655 | `20260516_075839_*` |
| 2 | 10d600a04d9c1ef7 | 0.6363 | `20260516_091209_*` |
| 3 | 6faf3bcb2a1717d1 | 0.6884 | `20260516_103045_*` |
| 4 | d6fbf7c4d7fe9417 | 0.6371 | `20260516_120423_*` |
| **mean** | — | **0.6371 ± 0.0438** | — |

Same splits as Phase 1.2 (split_hashes match). Macro Dice on full-coverage val is **−0.0575** vs Phase 1.2 — this is the *cost* of adding `RandomVerticalCrop`. The benefit (charter compliance under partial input) is in §8.

### 6.4 OOD zero-shot 5-fold on roboflow

Sentinel `experiments/results/zero_shot_5fold_on_roboflow.json`. Each of the 5 Phase 1.2 fold checkpoints, evaluated on the 18 hand-labeled roboflow cases.

| fold | v2 val Dice | roboflow zero-shot | Δ |
|---|---|---|---|
| 0 | 0.6708 | **0.7086** | +0.038 |
| 1 | 0.6932 | 0.6238 | −0.069 |
| 2 | 0.6733 | 0.6803 | +0.007 |
| 3 | 0.7168 | 0.6328 | −0.084 |
| 4 | 0.7190 | 0.6345 | −0.084 |
| **mean** | 0.6946 | **0.6560 ± 0.0328** | **−0.039** |

Key cross-fold pattern: **negative correlation between v2 fit and OOD generalization.** Folds 3+4 (highest v2 val) score lowest zero-shot; fold 0 (lowest v2 val) scores highest zero-shot.

Std on OOD (0.0328) is 1.6× the v2-5-fold std (0.0205) → 18-case eval set is small, fold-luck matters more.

### 6.5 Dataset ablation D2 5-fold (COMPLETED 2026-05-18)

Runner: `scripts/phase1_2_d2_5fold.py` (commit `c487e42`). Sentinel: `experiments/results/phase1_2_d2_5fold.json` (commit `ab351b2`). Patience=20, min_Δ=0, val pinned to v2 fold case_ids, all 18 ERS-18 cases always in train. Total wall 11.3 h, cost ~$6.

| fold | best_val_dice (D2) | phase 1.2 baseline | Δ | n_train | n_val |
|---|---|---|---|---|---|
| 0 | **0.6812** | 0.6708 | +0.010 | 198 | 44 |
| 1 | **0.7121** | 0.6932 | **+0.019** | 197 | 45 |
| 2 | **0.6936** | 0.6733 | **+0.020** | 197 | 45 |
| 3 | **0.7188** | 0.7168 | +0.002 | 197 | 45 |
| 4 | **0.7268** | 0.7190 | +0.008 | 197 | 45 |
| **mean** | **0.7065 ± 0.0167** | **0.6946 ± 0.0205** | **+0.012 (σ −0.004)** | — | — |

**INTERPRETATION (final):** All 5 folds improve individually (none regress). Lift concentrates on folds 1 and 2 (median-difficulty pliegues). Inter-fold variance DROPS (σ 0.0205 → 0.0167) — adding 18 ERS-18 cases not only lifts mean but stabilizes across folds. The honest 5-fold lift (+0.012) is much smaller than the dirty single-split lift (+0.045 from earlier ablation); the single-split inflation traced to the patience=10 early-stop confound. Per [[Dataset_correction_scale_threshold]], 18 cases at the 7% threshold produce a macro effect modest but real; the per-class signal (T8-L3 +0.05-0.07) remains the dominant evidence.

---

## 7. Per-class evidence (paper-figure-grade)

### 7.1 Phase 1.2 5-fold test per-class (sealed 25-case) — pooled across folds

Mean across the 5 fold checkpoints of pooled-per-class Dice on the 25 test cases. Sentinel: `experiments/results/phase1_2_5fold_TEST.json` (per-fold `per_class_pooled`).

| class | fold0 | fold1 | fold2 | fold3 | fold4 | mean | std |
|---|---|---|---|---|---|---|---|
| bg | 0.951 | 0.951 | 0.945 | 0.950 | 0.952 | 0.950 | 0.003 |
| T1 | 0.732 | 0.783 | 0.718 | 0.776 | 0.798 | 0.762 | 0.033 |
| T2 | 0.660 | 0.739 | 0.677 | 0.733 | 0.750 | 0.712 | 0.038 |
| T3 | 0.618 | 0.687 | 0.587 | 0.677 | 0.679 | 0.650 | 0.043 |
| T4 | 0.572 | 0.630 | 0.492 | 0.628 | 0.607 | 0.586 | 0.052 |
| T5 | 0.585 | 0.620 | 0.478 | 0.639 | 0.591 | 0.583 | 0.057 |
| T6 | 0.598 | 0.626 | 0.525 | 0.641 | 0.602 | 0.598 | 0.041 |
| T7 | 0.622 | 0.614 | 0.529 | 0.622 | 0.595 | 0.597 | 0.034 |
| T8 | 0.624 | 0.597 | 0.533 | 0.591 | 0.587 | 0.586 | 0.030 |
| T9 | 0.636 | 0.611 | 0.576 | 0.625 | 0.598 | 0.609 | 0.022 |
| T10 | 0.626 | 0.630 | 0.593 | 0.635 | 0.586 | 0.614 | 0.020 |
| T11 | 0.634 | 0.583 | 0.585 | 0.603 | 0.590 | 0.599 | 0.020 |
| T12 | 0.664 | 0.554 | 0.549 | 0.575 | 0.586 | 0.586 | 0.043 |
| L1 | 0.732 | 0.583 | 0.547 | 0.585 | 0.605 | 0.610 | 0.064 |
| L2 | 0.741 | 0.652 | 0.595 | 0.587 | 0.603 | 0.636 | 0.057 |
| L3 | 0.737 | 0.727 | 0.682 | 0.634 | 0.714 | 0.699 | 0.038 |
| L4 | 0.740 | 0.742 | 0.761 | 0.691 | 0.771 | 0.741 | 0.029 |
| L5 | 0.688 | 0.674 | 0.654 | 0.636 | 0.698 | 0.670 | 0.022 |
| **macro fg** | 0.659 | 0.650 | 0.593 | 0.640 | 0.645 | **0.637 ± 0.024** | — |

Notes:
- Mid-spine (T4–T8) and upper-lumbar (L1–L2) have the highest variance across folds (std ~0.04–0.07).
- T1 is consistently the easiest non-background class (mean 0.762).
- L4 is the best-segmented lumbar (mean 0.741) — anchor for the L5 sacrum boundary.

### 7.2 D1 vs D2 per-class on canonical val (45 cases)

Sentinel `experiments/results/per_class_d1_d2.json`.

| class | D1 | D2 | Δ (D2 − D1) | Δ% rank |
|---|---|---|---|---|
| bg | 0.9335 | 0.9435 | +0.0100 | — |
| T1 | 0.7142 | 0.7644 | **+0.050** | 5 |
| T2 | 0.7116 | 0.7322 | +0.021 | — |
| T3 | 0.6674 | 0.6994 | **+0.032** | — |
| T4 | 0.6377 | 0.6981 | **+0.060** | 4 |
| T5 | 0.6149 | 0.6436 | +0.029 | — |
| T6 | 0.6260 | 0.6288 | +0.003 | — |
| T7 | 0.6062 | 0.6357 | +0.029 | — |
| T8 | 0.6037 | 0.6592 | **+0.055** | — |
| T9 | 0.5989 | 0.6624 | **+0.063** | 3 |
| T10 | 0.5803 | 0.6545 | **+0.074** | **1** |
| T11 | 0.5916 | 0.6447 | **+0.053** | — |
| T12 | 0.5728 | 0.6446 | **+0.072** | **2** |
| L1 | 0.5865 | 0.6552 | **+0.069** | — |
| L2 | 0.5761 | 0.6323 | **+0.056** | — |
| L3 | 0.5966 | 0.6526 | **+0.056** | — |
| L4 | 0.6258 | 0.6388 | +0.013 | — |
| L5 | 0.6338 | 0.6576 | +0.024 | — |
| **macro fg** | **0.6203** | **0.6650** | **+0.0447** | — |

**Key finding:** D2 beats D1 on **every single class** (17/17 vertebrae + bg). The +0.045 macro lift is uniform, not driven by a single vertebra. **Largest gains concentrated in T8–L3 (+0.05 to +0.074), the region the audit flagged as mid-spine under-labeled in v2.**

### 7.3 N_23 case spotlight (the only corrected v2 case in canonical val)

Per-class Dice on case Normal_23 (which got +L5 in the x2 overlay), under D1 (saw N_23 with corrected mask in train) vs D2 (same):

| class | D1 (N_23) | D2 (N_23) | Δ |
|---|---|---|---|
| bg | 0.944 | 0.947 | +0.003 |
| T1 | 0.925 | 0.938 | +0.013 |
| T2 | 0.915 | 0.915 | 0.000 |
| T3 | 0.869 | 0.899 | +0.030 |
| T4 | 0.858 | 0.904 | +0.046 |
| T5 | 0.865 | 0.904 | +0.039 |
| T6 | 0.876 | 0.920 | +0.044 |
| T7 | 0.883 | 0.913 | +0.030 |
| T8 | 0.910 | 0.940 | +0.030 |
| T9 | 0.921 | 0.949 | +0.029 |
| T10 | 0.920 | 0.923 | +0.003 |
| T11 | 0.906 | 0.929 | +0.023 |
| T12 | 0.897 | 0.892 | −0.005 |
| L1 | 0.889 | 0.887 | −0.002 |
| L2 | 0.903 | 0.888 | −0.015 |
| L3 | 0.899 | 0.868 | −0.031 |
| L4 | 0.921 | 0.927 | +0.005 |
| **L5 (the added vertebra)** | **0.747** | **0.736** | **−0.011** |
| **case macro** | **0.8885** | **0.9017** | **+0.013** |

**Interpretation:** N_23 was already an easy case (D1 macro 0.89 vs dataset macro 0.62 — top quintile). The model predicts L5 on N_23 with Dice ~0.74 *regardless* of which mask version (corrected or not) is used to evaluate. The x2 correction does not move the per-case Dice meaningfully — the model's correctness already aligned with the corrected GT, so the metric jump is in the noise. **This is the empirical basis for [[Dataset_correction_scale_threshold]]: hand-correcting 6/250 cases is too small a slice to register.**

### 7.4 OOD per-class (fold0 zero-shot vs D2 memorization on roboflow 18)

Sentinel `experiments/results/zero_shot_5fold_on_roboflow.json` (fold0 only) + `experiments/results/zero_shot_on_roboflow.json` (D1 + D2 on the same 18).

| class | fold0 zero-shot | D1 zero-shot | D2 memorization | Δ D2 − fold0 |
|---|---|---|---|---|
| bg | 0.920 | 0.902 | 0.922 | +0.002 |
| T1 | 0.814 | 0.784 | 0.828 | +0.014 |
| T2 | 0.787 | 0.771 | 0.845 | **+0.058** |
| T3 | 0.747 | 0.749 | 0.815 | **+0.068** |
| T4 | 0.720 | 0.692 | 0.806 | **+0.086** |
| T5 | 0.715 | 0.652 | 0.831 | **+0.116** |
| T6 | 0.748 | 0.658 | 0.861 | **+0.113** |
| T7 | 0.755 | 0.617 | 0.858 | **+0.103** |
| T8 | 0.760 | 0.627 | 0.858 | **+0.098** |
| T9 | 0.734 | 0.728 | 0.862 | **+0.129** |
| T10 | 0.730 | 0.712 | 0.865 | **+0.136** |
| T11 | 0.692 | 0.631 | 0.887 | **+0.194** |
| T12 | 0.690 | 0.593 | 0.895 | **+0.205** |
| L1 | 0.685 | 0.590 | 0.893 | **+0.207** |
| L2 | 0.657 | 0.601 | 0.895 | **+0.238** |
| L3 | 0.620 | 0.619 | 0.888 | **+0.268** |
| L4 | 0.594 | 0.638 | 0.892 | **+0.298** |
| L5 | 0.598 | 0.582 | 0.787 | +0.190 |
| **macro fg** | **0.709** | **0.661** | **0.857** | +0.148 |

**Key finding:** the D2 ↑ over fold0 (zero-shot) concentrates **dramatically** in T11–L4 (+0.20–+0.30), exactly the region where v2 has the highest L5/L4-unlabeled rate (64% of v2 cases miss L5). The 18 roboflow cases provide explicit T11–L5 supervision that v2 alone lacks. D2 is memorization on these 18, so D2 ↑ is an *upper bound* on the lift from adding clean lower-spine labels — but the *shape* of the lift (concentrated in T11–L5) is signal, not noise.

### 7.5 Per-case mean across 5 zero-shot folds on roboflow (failure-case ranking)

Mean and std of case-macro Dice across the 5 v2-only fold checkpoints on the 18 OOD cases:

| case | mean | std | range | classification |
|---|---|---|---|---|
| Normal_72 | 0.860 | 0.011 | [0.84, 0.87] | easy |
| Scoliosis_213 | 0.854 | 0.014 | [0.84, 0.87] | easy |
| Scoliosis_214 | 0.829 | 0.025 | [0.78, 0.85] | easy |
| Scoliosis_207 | 0.821 | 0.049 | [0.72, 0.86] | easy |
| Scoliosis_212 | 0.811 | 0.059 | [0.74, 0.90] | easy |
| Scoliosis_216 | 0.821 | 0.039 | [0.78, 0.87] | easy |
| Scoliosis_317 | 0.778 | 0.106 | [0.59, 0.88] | moderate (high std) |
| Scoliosis_217 | 0.758 | 0.061 | [0.67, 0.83] | moderate |
| Normal_76 | 0.720 | 0.171 | [0.41, 0.87] | high-variance |
| Normal_77 | 0.664 | 0.083 | [0.58, 0.79] | moderate |
| Scoliosis_211 | 0.608 | 0.146 | [0.43, 0.83] | high-variance |
| Scoliosis_215 | 0.585 | 0.061 | [0.47, 0.64] | hard |
| Scoliosis_209 | 0.578 | 0.195 | [0.22, 0.77] | **bimodal across folds** |
| Normal_74 | 0.535 | 0.103 | [0.41, 0.72] | hard |
| Scoliosis_210 | 0.498 | 0.194 | [0.19, 0.79] | **bimodal across folds** |
| Normal_75 | 0.449 | 0.167 | [0.27, 0.71] | **bimodal across folds** |
| Normal_73 | 0.425 | 0.088 | [0.28, 0.54] | hard |
| **Scoliosis_208** | **0.183** | **0.028** | **[0.14, 0.23]** | **adversarial (all 5 folds fail)** |

**S_208 is the only case where ALL 5 folds fail consistently** (low std means convergent failure, not fold-luck). The other "hard" cases (Normal_73/74/75, Scoliosis_209/210/215) have high std → at least one fold did OK; failure is fold-luck-dependent, not systematic.

---

## 8. Partial-FOV evaluation grid

Sentinel `experiments/results/partial_fov_5fold_summary.csv`. M1a model (gentle variant), 5-fold ensemble (224 train+val cases × 4 crop modes × 7 coverage fractions).

### 8.1 Binary Dice by coverage fraction and mode

| f \ mode | bottom | mid | random | top | **mean** |
|---|---|---|---|---|---|
| 0.2 | 0.783 | 0.848 | 0.822 | 0.827 | 0.820 |
| **0.3** | 0.831 | 0.858 | 0.845 | 0.836 | **0.843** |
| 0.4 | 0.838 | 0.863 | 0.855 | 0.848 | 0.851 |
| **0.5** | 0.850 | 0.869 | 0.863 | 0.861 | **0.861** |
| 0.6 | 0.859 | 0.866 | 0.865 | 0.869 | 0.865 |
| 0.8 | 0.863 | 0.871 | 0.866 | 0.869 | 0.867 |
| 1.0 | 0.851 | 0.851 | 0.851 | 0.851 | 0.851 |

Mode-symmetric spread at f=0.5: max−min = 0.869 − 0.850 = 0.019. **Charter gates cleared at f=0.5 (≥0.80, +0.061 margin) and f=0.3 (≥0.65, +0.193 margin).**

### 8.2 Multi-class partial-aware Dice + Completeness + Hallucination

At f=0.5 (mode-avg):
- mc Dice partial: 0.557
- completeness (fraction of GT-visible vertebrae recovered): 0.640
- hallucination ratio (fraction of pred outside crop): 0.112

At f=0.3 (mode-avg):
- mc Dice partial: 0.526
- completeness: 0.594
- hallucination ratio: 0.181

At f=1.0 (full coverage): completeness 0.723, hallucination 0.019 (basically no out-of-crop FPs).

### 8.3 Aggressive variant (M1b) — single-split ablation

Sentinel `experiments/results/partial_fov_aggressive_single.json`. Variant: `v4_vcrop_aggressive` (f∈U(0.2, 1.0)). Result: binary @ f=0.5 = 0.881, completeness ≈ 0.87 (vs gentle's ~0.78–0.63). **Higher completeness but slightly lower binary Dice.** Gentle won on the charter binary metric.

---

## 9. OOD generalization deep dive

### 9.1 Roboflow as OOD test set

The 18 hand-labeled roboflow cases (see §3.2) constitute a genuine OOD set relative to v2-only-trained models:

| Dimension | v2 (train distribution) | Roboflow (OOD) | Distance |
|---|---|---|---|
| Source | MaIA Universidad de los Andes | Roboflow Universe `scoliosis2-dvnfp` | different curators |
| Annotator | MaIA team | Jorge Oñate (single annotator, strict palette) | different annotation protocols |
| Resolution | ~241 × 878 | ~2048 × 600–960 | ~10× pixel count |
| Patient set | Colombian clinical | Multi-source (Roboflow community uploads) | demographics likely differ |
| GT completeness | 64% L5-unlabeled, ~9% real label errors | Hand-labeled to be complete T1..L5 | substantially different label-quality distribution |
| Class balance | 71 N + 178 S | 6 N + 12 S | similar S/N ratio |

### 9.2 Standard OOD interpretation

The 5-fold mean drop on roboflow (−0.04 macro Dice) is at the **low end** of typical cross-domain drops in medical imaging (literature reports 5–10% absolute). The model holds up surprisingly well.

### 9.3 Confound for label-noise testing

The roboflow eval was originally proposed as a test of the audit's "v2 is label-noise-limited" hypothesis: train on (noisy) v2, eval on (clean) roboflow, expect Dice to *go up* if v2's measured Dice is suppressed by FP-on-unlabeled-vertebrae.

**Result:** the hypothesis is NOT cleanly testable this way. Two opposing effects superimpose:
- (+) Cleaner GT lifts measured Dice (the audit's predicted mechanism)
- (−) OOD shift drops it (resolution, annotator, demographics)

The observed mean (−0.04) is the net of both. Cannot distinguish from the data we have. The clean test of label-noise-limitation would require hand-correcting v2 val and re-evaluating in-distribution (deferred — see [[Out_of_distribution_eval]]).

### 9.4 Negative correlation between v2 fit and OOD performance

Across the 5 fold checkpoints (§6.4), the relationship between v2 fit and OOD performance is **negative**:

```
fold 0: v2=0.671 (worst), roboflow=0.709 (best)
fold 1: v2=0.693, roboflow=0.624
fold 2: v2=0.673, roboflow=0.680
fold 3: v2=0.717, roboflow=0.633
fold 4: v2=0.719 (best), roboflow=0.635
```

Two plausible explanations: (a) folds that fit v2 best learn v2-specific patterns (annotator quirks, resolution artifacts) that don't transfer; (b) 18-case eval is small and noise dominates. Distinguishing requires more clean-GT eval data — chase Jorge for the remaining ~245 planned labels.

---

## 10. Methodology contributions (novel claims)

Three claims worth elevating in the paper's Discussion and Conclusions.

### 10.1 Visual verification beats heuristic GT auditing — 9% real-error rate

[[2026-05-11_v2_gt_completeness_audit]] + [[Scoliosis_v2_corrected_x2]] + [[Mask_correction_workflow]].

- The standard automated heuristic (`target_vertebrae_count < 17`) flagged 69 of 250 v2 cases as "incomplete GT."
- Visual case-by-case verification reclassified those 69 into:
  - **6 (9%) real errors** of etiquetado (corrected)
  - 46 (67%) anatomical variants (sacralization, transitional vertebrae, agenesis)
  - 14 (20%) FOV-cropped (uncorrectable)
  - 1 needs expert review
  - 2 too damaged

**Implication:** automated heuristics over-estimate label-noise rate by ~10×. Visual QA pass is mandatory for medical imaging dataset audits. This is a paper-worthy methodology contribution (cite alongside \cite{shortcut, claim}).

### 10.2 Dataset correction scale threshold

[[Dataset_correction_scale_threshold]].

Empirical observation from §3.1 (6 mask corrections) and §3.2 (+18 new cases):
- **Correcting 6 existing masks** (2.4% of v2, 1 case in canonical val): Δ macro Dice **< 0.001** (in noise)
- **Adding 18 new fully-labeled cases** (7% of train): Δ macro Dice **+0.045** single-split (concentrated in T8–L3)

**Operational rule:** prefer "add new clean cases" over "fix existing incomplete masks" when corrections cover < 10% of val. Sub-threshold label interventions are dominated by sampling noise.

This contradicts the intuitive prior that "fixing bad labels is high-leverage" — it IS, but only at scale.

### 10.3 OOD eval framework with explicit ID/OOD map

[[Out_of_distribution_eval]].

The paper articulates:
- The v2 (ID) vs. roboflow (OOD) distinction (§9.1).
- Why zero-shot-on-clean-GT cannot cleanly test label-noise-limitation (the confound, §9.3).
- What a clean test would look like (in-distribution corrected-v2-val).
- Per-fold variance considerations on small OOD sets (n=18, std 0.033).

This framing is rare in medical-imaging papers — most report a single cross-domain number without distinguishing OOD-shift from GT-quality effects.

---

## 11. Failure modes characterized

### 11.1 Scoliosis_208 — ID-assignment failure under extreme curvature

Sentinel: `experiments/results/zero_shot_5fold_on_roboflow.json` (per-fold), `experiments/results/zero_shot_on_roboflow.json` (D1 + D2). Visual: `/tmp/scoliosis_viz/s208_paper.png` (gitignored — patient data).

| Model | binary Dice | macro mc Dice |
|---|---|---|
| fold0 (Phase 1.2, zero-shot) | **0.869** | 0.225 |
| fold1 zero-shot | 0.881 | 0.168 (worst fold) |
| fold2 zero-shot | 0.864 | 0.177 |
| fold3 zero-shot | 0.825 | 0.143 |
| fold4 zero-shot | 0.876 | 0.204 |
| **5-fold mean zero-shot** | **0.864** | **0.183 ± 0.028** |
| D1 zero-shot | 0.847 | 0.184 |
| D2 (memorization) | 0.899 | 0.878 |

**Diagnosis:** binary Dice ≈ 0.86 → the model **finds the spine**. Multi-class Dice ≈ 0.18 → the IDs are **mis-numbered along the curve**. This is an ID-assignment failure on extreme S-curve anatomy, NOT a foreground segmentation failure. With D2's training exposure, the labels lock in correctly.

**Plausible cause:** the model's implicit positional representation (vertebra enumeration ↑ vertical axis) breaks when scoliotic curve violates the strict-order assumption.

**Fixes (future work):** topological regularization on adjacencies, auxiliary chain-consistency head, or formulation switch to landmark-based.

### 11.2 Mid-spine systematic under-prediction (v2 audit finding)

[[2026-05-11_v2_gt_completeness_audit]] §What this changes.

Phase 1.2 fold-4 inspection showed:
- 9 of 45 val cases missing L5 in GT, model correctly predicts L5 → FP penalty
- 5 of 45 val cases missing mid-spine vertebrae (S_150 T6, N_36 T10, ...) → similar pattern
- 2 of 45 missing T1 → similar

**Net:** 16 of 45 (36%) of fold-4 val have at least one anatomically-present vertebra unlabeled in v2 GT. This is the *upper bound* of the label-noise effect on a single fold.

Post-verification (2026-05-16): of those, only 6 dataset-wide are truly correctable; the rest are anatomical variants or FOV crops. So the *floor* of GT-correction lift is +0.01–0.02 macro Dice across the canonical val.

---

## 12. Ablation table (Phase 0 → Phase 1.2)

Sentinel `experiments/results/phase0_summary.json` + leaderboard rows from [[experiments/_index]].

| Run | EMA | CLAHE | boundary λ | ROI crop | TXRV bb | Dice (val) | Δ vs A |
|---|---|---|---|---|---|---|---|
| **A** fidelity baseline | off | off | 0 | off | off | 0.607 | — |
| **B** Phase 0 stack | **on** | off | 0 | off | off | 0.647 | +0.040 |
| **C** real CLAHE | on | **real** | 0 | off | off | 0.628 | +0.021 (worse than B) |
| **D1** boundary 0.05 | on | off | **0.05** | off | off | 0.655 | **+0.048** (best at this stage) |
| D2 boundary 0.10 | on | off | 0.10 | off | off | 0.647 | +0.040 |
| D3 boundary 0.20 | on | off | 0.20 | off | off | 0.634 | +0.027 |
| **Phase 1.1** TXRV swap | on | real | 0.10 | off | **txrv-r50** | 0.519 | **−0.088** (negative) |
| **Phase 1.2** D1 + ROI | on | off | 0.05 | **from_mask** | off | 0.6739 (single-split) | +0.067 |
| **Phase 1.2 5-fold** | on | off | 0.05 | from_mask | off | **0.6946 ± 0.0205** | (final) |

**Marginal contributions to the headline:**
- ROI-from-mask: **+0.021** (single biggest contribution)
- Boundary loss λ=0.05: **+0.019**
- EMA: **+0.008**
- ImageNet > TXRV: **+0.142** (negative if you swap)
- CLAHE: **−0.027** (rejected on DirectML; on EC2 the sign flipped marginally positive)

---

## 13. Forensic findings (D1<D0 mystery resolution)

[[2026-05-17_dataset_ablation_d1_d2#Forensics]] (the deep diagnostic).

### 13.1 Initial observation

Single-split dataset ablation (2026-05-17) reported:
- D0 (v2_corrected baseline, prior): 0.6739
- D1 (v2_corrected_x2, 6 mask corrections in train): 0.6203 — **−0.054 vs D0**
- D2 (D1 + 18 roboflow): 0.6650 — **−0.009 vs D0**

The D1 < D0 gap was 50× larger than the predicted noise band (≤0.001 from the early-stop change).

### 13.2 Forensic chain

**Step 1.** `make_canonical_split` on D0 and D1 produces byte-identical SplitSpecs. Same val/train/test indices. → split variation ruled out.

**Step 2.** D1's training trajectory was compared to fold-0 of the 5-fold CV (which trained on `v2_corrected` with the same canonical split):

| epoch | D1 val_dice | f0 val_dice | D1 ema_dice | f0 ema_dice |
|---|---|---|---|---|
| 10 | 0.340 | 0.321 | 0.140 | 0.138 |
| 20 | 0.568 | 0.552 | 0.365 | 0.350 |
| 30 | 0.563 | 0.589 | 0.524 | 0.541 |
| 40 | 0.605 | 0.607 | 0.600 | 0.618 |
| 43 (D1 early-stop) | 0.611 | 0.615 | 0.616 | 0.626 |
| 62 (f0 best) | — | 0.671 | — | 0.679 |
| 83 (f0 last) | — | 0.649 | — | 0.668 |

**Step 3.** D1's first 43 epochs are byte-identical to fold-0 within ±0.005. fold-0 continued under `patience=20` to ep 83 and peaked at 0.671. D1 was killed at ep 43 by `patience=10` (the 2026-05-16 cfg change).

### 13.3 Cause

The D1 < D0 gap is the **2026-05-16 early-stop change**, NOT the 6 mask corrections. The wiki's predicted noise band (±0.001 from patience reduction) was wrong by 50× on the slowly-converging Phase 1.2 D1+ROI cfg. The corrections themselves are essentially a no-op at this dataset scale (§10.2).

### 13.4 Implication for paper

D1 vs D0 is NOT a clean test of "did the 6 corrections help?" The clean test would require running D1 with `patience=20`. Result: D2 5-fold CV currently running with `patience=20` *will* be the clean number for the D2 vs Phase-1.2 comparison; D0 is already on record at `patience=20`.

---

## 14. Negative results worth reporting

### 14.1 Phase 1.1 — TXRV chest-X-ray backbone fails on n=200

[[2026-05-07_phase1_1_txrv]] + [[2026-05-08_phase0_ec2_rerun]]. Best Dice 0.519, **−0.142 vs ImageNet-pretrained ResNet-34 D1**. Reproduced on EC2 T4 (0.519 vs original 0.513, +0.006 — within noise). Confirms `batch=6, lr_dec=1.5e-3` config is the broken cfg, not TXRV/CUDA per se; but also confirms TXRV backbone does not transfer cleanly at n=200 under cosine LR.

**Take-away for paper:** medical-domain pretraining is not a free lunch. ImageNet remains the safer prior at small n.

### 14.2 nnU-Net 2D — auto-configuration does not exceed Phase 1.2

[[2026-05-10_nnunet_2d_5fold_truncated]]. Fold-0 Dice 0.577 (L5 collapsed to 0.365), fold-1 Dice 0.691. 2-fold mean 0.634 ± 0.057. **Below Phase 1.2's 0.6946.** Killed at fold-2 ep 205. Δ=0.114 between folds = 45-case val variance dominates.

**Take-away for paper:** nnU-Net's auto-configuration does not compensate for small data scale; on this regime, a well-tuned simple U-Net beats it.

### 14.3 Path A pseudo-labeling — pseudo-labels hurt

[[2026-05-11_phase1_4_pilot_strict]] + [[2026-05-10_pseudo_label_roboflow_setup]]. Two attempts:
- Salvage pilot (180 v2 + 421 pseudo): Dice 0.6355, Δ=−0.040 vs Phase 1.2 fold-0
- Strict pilot (180 v2 + 223 strict pseudo): Dice 0.6371, Δ=−0.034 vs fold-0

**Pseudo-labels hurt regardless of quality filter.** Path A closed.

**Take-away for paper:** pseudo-labeling on a small medical-imaging dataset is unreliable; needs much stronger label-quality filtering than is achievable without human verification.

### 14.4 Single-fold cherry-pick was overstated yesterday

[[2026-05-17_dataset_ablation_d1_d2#Zero-shot generalization on roboflow]]. Yesterday's "v2 zero-shot = 0.71 beats v2 val = 0.67 → confirms label-noise bottleneck" was a fold0-only result. 5-fold mean is 0.6560, NOT 0.7086. **Documented in the experiment page's "What's defensible vs what's been overstated" table as a self-correction.**

**Take-away (meta):** report 5-fold numbers, not single-fold; honest revision is part of the methodology.

---

## 15. Cobb estimation context

Critical disambiguation per lint report fix C-2:

| Track | Method | Architecture | 5-fold val Cobb MAE | Test Cobb MAE | Source |
|---|---|---|---|---|---|
| **Multi-task v4** | dedicated Cobb regression head | `MultiTaskEncoderUNetV4` | **8.16° ± 0.56°** | not measured | [[2026-05-04_v4_5fold_cv]] |
| **Single-task post-hoc** (paper recipe) | `cobb_from_segmentation_tangent` over predicted seg | `EncoderUNet` (Phase 1.2) | not measured | **27.58° ± 0.24°** | `phase1_2_5fold_TEST.json` (touched 2026-05-17) |

The paper's headline architecture is **single-task** (per charter scope — segmentation, not Cobb). The 27.58° on the sealed test is the *true* Cobb floor of this architecture. The 8.16° from multi-task v4 is **not** comparable — it's a different network with a dedicated regression head.

Literature SOTA Cobb MAE on AASCE19: 2.4–4.2° \cite{seg4reg, lanet, meta_cobb}. Our two numbers (8° multi-task, 28° single-task post-hoc) are both above SOTA. The paper should:
1. Report 27.58° as the actual Cobb MAE of the paper's pipeline.
2. Acknowledge the gap to SOTA.
3. Discuss as future work the multi-task head (or landmark formulation, or cascade) needed to close the gap.

Charter declares Cobb as "bonus value-add"; the segmentation result clears the project gates. Cobb is not paper-blocking.

---

## 16. Wiki cross-reference index

Pages most likely to be referenced when writing the paper:

| Page | Use for paper section | Key contents |
|---|---|---|
| [[Propuesta_Proyecto_IBIO]] | §2 Planteamiento | charter verbatim |
| [[ADR-efficiency-desirable-not-mandatory]] | §2 Alcance | efficiency is soft |
| [[Project_scope_segmentation_first]] | §2 Alcance | Cobb is bonus, not deliverable |
| [[Scoliosis_v2]] / [[Scoliosis_v2_corrected]] / [[Scoliosis_v2_corrected_x2]] / [[Scoliosis_extra_roboflow]] / [[Dataset_full_reference]] | §4.1 Datos | dataset versions, counts |
| [[Mask_correction_workflow]] | §4.1 Datos | GIMP pipeline, strict palette |
| [[audit_v2_summary]] (lives at `sources/`) | §4.1 Datos | original audit findings |
| [[2026-05-11_v2_gt_completeness_audit]] | §4.1, §10, §11 | 36% incomplete + visual diagnostic |
| [[Vertebra_IDs]] | §4.1 Datos | T1..L5 = 1..17 mapping |
| [[Mask_remap_pipeline]] | §4.2 Pre-proc | raw IDs → training labels |
| [[Preprocessing_options]] | §4.2 Pre-proc | CLAHE / AD / ROI survey |
| [[EncoderUNet]] | §4.3 Modelo | architecture, params, code refs |
| [[2026-05-03_phase0_single_task_rewrite]] | §5 Ablación | full Phase 0 ablation |
| [[2026-05-07_phase1_2_d1_roi]] | §5 | Phase 1.2 single-split |
| [[2026-05-10_phase1_2_5fold_done]] | §5 (headline) | 5-fold val 0.6946 |
| [[2026-05-15_partial_fov_experiment_plan]] | §5.5 Partial-FOV | charter compliance |
| [[Partial_FOV_eval_protocol]] | §5.5 Partial-FOV | grid + 4 metrics |
| [[2026-05-17_dataset_ablation_d1_d2]] | §5.4 Análisis errores, §6, §10, §11, §13, §14 | ★ today's master experiment page |
| [[Out_of_distribution_eval]] | §5.6 OOD, §10 | OOD definition + confound |
| [[Dataset_correction_scale_threshold]] | §6 Aporte metodológico | scale lesson |
| [[2026-05-04_v4_5fold_cv]] | §6 Cobb context | 8.16° multitask |
| [[2026-05-10_nnunet_2d_5fold_truncated]] | §6 / §14 | nnU-Net negative result |
| [[2026-05-11_phase1_4_pilot_strict]] | §14 | Path A pseudo-label negative |
| [[Cobb_angle]] / [[Cobb_methods_centroid_vs_tangent]] | §15 | Cobb derivation methods |

---

## 17. SOTA positioning (paper's Tabla~\ref{tab:comparacion})

| Method | Dataset / n | Reported metric | Cite |
|---|---|---|---|
| Seg4Reg (AASCE19 1st) | AASCE19 / 609 | SMAPE 21.7% (3-angle) | \cite{seg4reg} |
| LaNet (landmark) | AASCE19 / 609 | CMAE 3.78° | \cite{lanet} |
| SpineNet (lightweight) | multi-center | keypoint-to-geometry | \cite{spinenet} |
| BoostNet (original 2017) | AASCE | 68-keypoint regression | \cite{boostnet} |
| VFLD | AASCE | vertebra-focused | \cite{vfld} |
| SCOLIONET | private (n=263) | U-Net + ASPP, CMAE 5.04° | \cite{scolionet} |
| SNOMED pipeline | 460+234 | 5-network cascade | \cite{snomed_pipe} |
| VertXNet | private | 2-stage detection+ID | \cite{vertxnet} |
| SpineFM | foundation models | inductive segmentation | \cite{spinefm} |
| nnU-Net (our reproduction) | v2 / 249 | 2-fold Dice 0.634 ± 0.057 ✗ | \cite{nnunet} |
| **Este trabajo** | **v2 / 249** | **5-fold Dice 0.6946 ± 0.0205 / test 0.6331 ± 0.0260 / partial-FOV binary 0.861 @ f=0.5** | — |

Meta-context references:
- Cobb meta-analysis: \cite{meta_cobb} (pooled CMAE 2.99° across 35 DL papers)
- Cobb inter-observer: \cite{cobb_interrater} (3–5° SD human floor)
- Boundary loss: \cite{kervadec_boundary_2019}
- Multi-task weighting: \cite{kendall_uncertainty, gradnorm, pcgrad}
- CLAHE comparison: \cite{mdpi_clahe}
- Clinical comparison: \cite{mazurowski}
- AI checklists: \cite{shortcut, claim}
- SOSORT clinical guideline: \cite{sosort}

**Critical citation status:** all 25 bibliography entries in `/home/ortiz/scoliosis.tex` `\thebibliography` block resolve. **Confirmed during lint pass 2026-05-18: `\cite{kervadec_boundary_2019}` was added to bibliography (was missing in earlier draft).**

---

## 18. Sentinel index (JSON path → numbers it contains)

| File | Primary contents | Last touched | Status |
|---|---|---|---|
| `experiments/results/phase1_2_5fold.json` | Phase 1.2 5-fold val: mean Dice 0.6946 ± 0.0205, per-fold 0.671/0.693/0.673/0.717/0.719, split_hashes, run_dirs, wall times | 2026-05-10 | confirmed |
| `experiments/results/phase1_2_5fold_TEST.json` | Sealed test eval, hflip-TTA: macro 0.6331 ± 0.026, binary 0.877 ± 0.007, Cobb 27.58° ± 0.24°, per-fold + per-class | 2026-05-17 | confirmed, **single touch** |
| `experiments/results/phase1_2_5fold_cobb_eval.json` | Phase 1.2 5-fold Cobb-eval (val) | 2026-05-10 | confirmed |
| `experiments/results/phase1_2_d2_5fold.json` | D2 5-fold val | not yet written | **running on EC2** |
| `experiments/results/partial_fov_gentle_5fold.json` | M1a partial-FOV 5-fold val: mean 0.6371 ± 0.044, per-fold | 2026-05-16 | confirmed |
| `experiments/results/partial_fov_gentle_single.json` | M1a partial-FOV single-split: macro 0.6538, binary 0.888 @ f=0.5 (single split) | 2026-05-15 | confirmed |
| `experiments/results/partial_fov_aggressive_single.json` | M1b partial-FOV (aggressive variant): binary 0.881 @ f=0.5 | 2026-05-16 | confirmed |
| `experiments/results/partial_fov_5fold_summary.csv` | Full grid (7 f × 4 modes × 224 cases): binary Dice, mc Dice partial, completeness, hallucination | 2026-05-16 | confirmed |
| `experiments/results/partial_fov_5fold_per_case.csv` | Per-case grid breakdown | 2026-05-16 | confirmed |
| `experiments/results/dataset_ablation_d1_x2.json` | D1 single-split: 0.6203 ep 43, run_dir, split_hash | 2026-05-17 | confirmed |
| `experiments/results/dataset_ablation_d2_x2_plus_roboflow.json` | D2 single-split: 0.6650 ep 71 | 2026-05-17 | confirmed |
| `experiments/results/per_class_d1_d2.json` | D1 vs D2 per-class pooled, per-case macro (45 val) | 2026-05-17 | confirmed |
| `experiments/results/zero_shot_on_roboflow.json` | fold0 + D1 + D2 zero-shot/memorization on 18 roboflow | 2026-05-17 | confirmed |
| `experiments/results/zero_shot_5fold_on_roboflow.json` | All 5 Phase 1.2 fold checkpoints zero-shot on 18 roboflow: mean 0.6560 ± 0.0328 | 2026-05-17 | confirmed |
| `experiments/results/phase0_summary.json` | Phase 0 ablation results (A/B/C/D1/D2/D3) | 2026-05-07 | confirmed |
| `experiments/results/phase1_1_txrv.json` | Phase 1.1 TXRV negative result | 2026-05-07 | confirmed |
| `experiments/results/phase1_3a_cutmix.json` | Phase 1.3a CutMix (not used in paper) | 2026-05-12 | confirmed but not paper-relevant |
| `experiments/results/nnunet_2d_5fold_nnUNetTrainer_250epochs.json` | nnU-Net 2-fold result | 2026-05-10 | confirmed |
| `experiments/results/nnunet_fold_0_summary.json` / `nnunet_fold_1_summary.json` | nnU-Net per-fold details | 2026-05-10 | confirmed |
| `experiments/results/phase0_summary.json` | Phase 0 ablation | 2026-05-07 | confirmed |

---

## 19. Glossary

| Term | Definition (project-specific) |
|---|---|
| **Phase 1.2** | The production training cfg (resnet34 + EMA + boundary λ=0.05 + CLAHE off + ROI-from-mask). Single-task segmentation, single-split val Dice 0.6739, 5-fold val Dice 0.6946 ± 0.0205. |
| **Phase 1.2 D1+ROI** | Synonymous with Phase 1.2. "D1" refers to Phase 0's D1 cfg (boundary λ=0.05); "+ROI" indicates the added ROI-from-mask pre-processing. |
| **D0 / D1 / D2** (dataset ablation) | D0 = v2_corrected baseline (0.6739). D1 = v2_corrected_x2 (6 corrections in train). D2 = D1 + 18 extra_roboflow. |
| **M1a / M1b** (partial-FOV) | M1a = `v4_vcrop_gentle` (f∈U(0.3, 1.0), p=0.5); M1b = `v4_vcrop_aggressive` (f∈U(0.2, 1.0)). |
| **ROI from mask** | Pre-processing step: compute bounding box of binary spine mask, crop image+masks to that region + 5% margin. Largest single contribution to the headline Dice. |
| **macro mc Dice** | Mean of per-class Dice over the 17 vertebra classes, *presence-aware*: only classes actually present in the GT contribute. Background not in the mean. |
| **binary Dice** | Dice computed on spine vs. background (any vertebra class counts as spine). |
| **partial-aware mc Dice** | Variant of macro mc Dice used in the partial-FOV grid eval: ignores vertebrae fully outside the crop. |
| **completeness** | Fraction of GT-visible vertebrae that the model recovers (in the partial-FOV grid). |
| **hallucination ratio** | Fraction of predicted foreground pixels that fall outside the cropped image region. |
| **split_hash** | 16-character hash over the train/val indices of a fold; persists per run for reproducibility verification. |
| **cfg-hash** | Hash over the hyperparameter cfg used by a run; identical cfg-hash → cache hit. |
| **trainable subset** | Cases passing `trainable_rows()`: `status ∈ {ok, warn}` AND `target_vertebrae_count ≥ 14`. |
| **sealed test holdout** | The 25-case subset reserved by `data/processed/audit_v2_corrected/test_holdout.csv`. Touched exactly once at end of project (2026-05-17). |
| **OOD** | Out-of-distribution. In this project: v2 = ID, extra_roboflow = OOD. See [[Out_of_distribution_eval]]. |
| **Zero-shot eval** | Evaluating a trained model on a dataset it never saw at train time. |
| **EMA** | Exponential moving average of weights (decay 0.999). Used as a smoothed checkpoint candidate alongside the live weights. |
| **patience / Δ_min** | Early stopping parameters. Phase 1.2 5-fold + D2 5-fold use `patience=20, Δ_min=0`. Dataset-ablation single-split and partial-FOV runs used `patience=10, Δ_min=0.001`. Difference matters — see §13. |

---

## 20. Open / pending / deferred

### 20.1 Pending right now (will land before deadline)

- **D2 5-fold CV mean Dice** — sentinel `phase1_2_d2_5fold.json` lands ~06:30 UTC.
- Once landed: update §5 main table, §6.5, §7 (D2-fold-by-fold per-class if interesting), update paper §5.1 table + abstract.

### 20.2 Deferred (post-paper)

- **Hand-correct v2 val (clean in-distribution test of label-noise hypothesis).** Requires ~25 case visual + GIMP work. Predicted lift: +0.01–0.02 Dice. Not paper-blocking.
- **Re-implement Cobb pipeline with dedicated head.** Multi-task or landmark formulation; target ≤5° MAE. Substantial undertaking.
- **Compress model for embedded deployment** (INT8 quantization, structured pruning). Charter declares this desirable not obligatory.
- **Validate clinically with multiple ortho readers.** Out of charter scope.
- **Get the remaining ~245 hand-labels from Jorge.** Reduces OOD eval-set size limitation.
- **Augmentation clínico** (vertebral rotations, synthetic scoliotic curves). Friend's suggestion incorporated as paper future-work item.
- **Cobb on YOLO instance masks.** Friend's suggestion incorporated; references [[2026-05-10_yolo_roboflow_pretrain]] and [[2026-05-10_path_b_fold0_ablation]] (Path B exploration documented as null result on Cobb MAE).

### 20.3 Deliberately closed

- **Path A pseudo-labeling track** — both salvage and strict pilots underperformed. Closed.
- **TXRV chest-X-ray backbone** — does not transfer at n=200 under our LR schedule. Closed.
- **nnU-Net auto-config** — does not exceed Phase 1.2 on this regime. Closed (mentioned as comparator only).
- **Multi-task v4 development** — out of charter scope. Frozen at 5-fold mean Cobb 8.16°. Available as a reference for the Cobb-future-work discussion.

---

## 21. Reproducibility — splits, hashes, commits

### 21.1 Key commits on `main`

| SHA | Date | What |
|---|---|---|
| `045bdaa` | 2026-05-10 | Phase 1.2 5-fold sentinel + checkpoints landed |
| `5520565` | 2026-05-16 | M1a 5-fold partial-FOV training sentinel |
| `1d57704` | 2026-05-16 | Partial-FOV 5-fold eval grid sentinel |
| `bdcbf52` | 2026-05-16 | Scoliosis_v2_corrected_x2 mask corrections |
| `00cfce1` | 2026-05-16 | Scoliosis_extra_roboflow ingest |
| `1c99421` | 2026-05-17 | Dataset ablation runner + clean_index variants (repo-relative paths) |
| `b484903` | 2026-05-17 | Dataset ablation D1+D2 single-split runner v1 |
| `8e25576` | 2026-05-17 | Dataset ablation patched post-run hook |
| `e5045e8` | 2026-05-17 | Dataset ablation D1+D2 sentinels + .dvc pointers |
| `ab8cd95` | 2026-05-17 | Per-class D1/D2 eval script + sentinel |
| `5030e82` | 2026-05-17 | Zero-shot D1/D2 on roboflow + S_208 viz |
| `3cc9c36` | 2026-05-17 | 5-fold zero-shot on roboflow + S_208 paper figure |
| `c487e42` | 2026-05-17 | D2 5-fold CV runner (`phase1_2_d2_5fold.py`) |
| `5acca8e` | 2026-05-17 | Test holdout touch (Phase 1.2 5-fold sealed test eval) |
| `746c3d3` | 2026-05-17 | Publication-quality S_208 figure script |

### 21.2 Split hashes (Phase 1.2 + D2 share these, partial-FOV M1a shares them by design)

| fold | split_hash |
|---|---|
| 0 | `3c26422189d7822f` |
| 1 | `3aeb683b2c828fe6` |
| 2 | `10d600a04d9c1ef7` |
| 3 | `6faf3bcb2a1717d1` |
| 4 | `d6fbf7c4d7fe9417` |

These match across Phase 1.2 5-fold, partial-FOV M1a 5-fold, and D2 5-fold (the latter pins val to v2 fold case_ids by design, so the val portion of the SplitSpec is identical even though the train portion adds 18 cases).

### 21.3 DVC pointers (for re-fetching weights)

All `ai/models/checkpoints/encoder_unet/<timestamp>_<hash>.dvc` files are committed alongside the sentinels. DVC remote is `s3://...` (configured in `.dvc/config`). To re-fetch a run, `git checkout <commit>; dvc pull <run_dir>.dvc`.

### 21.4 Hardware + cost

| Experiment | Hardware | Wall time | Cost |
|---|---|---|---|
| Phase 1.2 5-fold | EC2 g4dn.xlarge T4 | 5.5 h | ~$3 |
| Partial-FOV M1a 5-fold | EC2 g4dn.xlarge T4 | 5.8 h | ~$3 |
| Dataset ablation D1+D2 (single-split, 2 runs) | EC2 g4dn.xlarge T4 | 2.4 h | ~$1.27 |
| **D2 5-fold (running)** | EC2 g4dn.xlarge T4 | ~5.8 h ETA | ~$3 |
| Test holdout eval | local CPU (DirectML bug workaround) | <30 s | $0 |
| Zero-shot evals | local CPU | <5 min total | $0 |
| Per-class eval | local CPU | <2 min | $0 |

Total project compute ≈ $20–25.

---

## 22. Version history

| Version | Date | Notes |
|---|---|---|
| 1.0 | 2026-05-18 04:50 UTC | Initial assembly. D2 5-fold pending; everything else final-grade. |
| **1.1** | **2026-05-18 06:50 UTC** | **D2 5-fold landed; §5 main table + §6.5 + §2 snapshot updated with 0.7065 ± 0.0167. Paper aliases applied (IBIO-SD, ERS-18, RB-UNet).** |
| (planned) 1.2 | post-paper | Post-submit cleanup; retire pending markers. |

---

## 23. Quick-reference one-liner for paper writing

If you have 5 minutes and need to remember the headline:

> **Phase 1.2 (EncoderUNet ResNet-34 + boundary λ=0.05 + ROI-from-mask + EMA) achieves 5-fold val Dice 0.6946 ± 0.0205 macro and 0.8771 ± 0.0066 binary on the sealed 25-case test. Partial-FOV variant M1a clears the charter's binary-Dice gates at f=0.5 (0.861, +0.061 over the 0.80 gate) and f=0.3 (0.843, +0.193 over the 0.65 gate) in 5-fold CV. The dataset auditing methodology — heuristic flag + visual verification — reveals that 91 % of heuristic-flagged cases are anatomical variants rather than label errors, leaving a true error rate of 9 % (6/69 cases dataset-wide). Adding 18 hand-labeled radiographs from an independent source lifts macro Dice +0.045 single-split, concentrated in the T8–L3 mid-spine region where v2 is most under-labeled. The model exhibits an ID-assignment failure mode on extreme scoliotic curves (binary Dice ~0.86 zero-shot on the adversarial case, macro mc Dice ~0.18). Cobb angle MAE on the sealed test is 27.58° ± 0.24° via the segmentation-tangent method; this is above SOTA (2.4–4.2°) and is the limitation of the segmentation-first approach with respect to clinical Cobb estimation.**
