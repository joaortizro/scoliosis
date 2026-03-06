# Notebooks

Three-tier workflow: sandbox → experiments → final.

```
notebooks/
├── sandbox/          # Personal scratchpads — no rules, no reviews
├── experiments/      # Shared topic notebooks — document what worked
│   ├── preprocessing/
│   ├── augmentation/
│   ├── architectures/
│   └── evaluation/
└── final/            # Thesis-ready — clean, reproducible, reviewed
```

---

## Rules

### `sandbox/`
- Name your file however you want: `jose_trying_clahe.ipynb`, `quick_test.ipynb`
- No code quality requirements
- Not reviewed by others
- Can be gitignored per-person if desired

### `experiments/`
- One notebook per idea/topic, named descriptively: `augmentation_mixup_vs_cutmix.ipynb`
- Must include a **Conclusions** cell at the bottom summarizing what worked and what didn't
- Promoted here from sandbox when an idea is worth sharing with the team

### `final/`
- Numbered sequentially: `01_eda.ipynb`, `02_preprocessing.ipynb`, ...
- Must be fully reproducible top-to-bottom (Kernel → Restart & Run All)
- These feed directly into the thesis
- Require team review before merging

---

## Promotion flow

```
sandbox/my_idea.ipynb
    ↓  (worth sharing?)
experiments/architectures/resnet_vs_efficientnet.ipynb
    ↓  (confirmed, clean, thesis-ready?)
final/03_modeling.ipynb
```
