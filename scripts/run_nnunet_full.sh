#!/bin/bash
# Morning launch: full nnU-Net 2D 5-fold on Dataset002_Spine (test-holdout-excluded).
# Uses 250-epoch trainer (compromise vs nnU-Net's default 1000) — convergence
# expected by ~150-200 ep on this dataset size.
#
# Usage on the g6e box (after `aws ec2 start-instances` + git pull origin iteration4):
#     bash scripts/run_nnunet_full.sh [--smoke]   # --smoke = 5 epochs only
#
# After all 5 folds complete, runs nnUNetv2_find_best_configuration to emit
# a cross-fold mean Dice number, dvc-adds + dvc-pushes the results dir,
# git-pushes the sentinel, and shutdown -h +2.
set -e

SMOKE=""
if [ "${1:-}" = "--smoke" ]; then
    SMOKE="--smoke"
    TRAINER="nnUNetTrainer_5epochs"
    EXPECTED_WALL="~5 min"
    EXPECTED_COST="~$0.20"
else
    TRAINER="nnUNetTrainer_250epochs"
    EXPECTED_WALL="~12-25 h"
    EXPECTED_COST="~$25-50"
fi

cd ~/scoliosis

export nnUNet_raw=~/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=~/nnunet_data/nnUNet_preprocessed
export nnUNet_results=~/nnunet_data/nnUNet_results
export nnUNet_n_proc_DA=0   # Python 3.13 + batchgenerators MP workaround
export PYTHONPATH=.

DATASET_ID="002"
DATASET_NAME="Spine"
DS_TAG="Dataset${DATASET_ID}_${DATASET_NAME}"

echo "[$(date -u +%FT%TZ)] === nnU-Net 2D 5-fold launch ==="
echo "trainer=${TRAINER} expected_wall=${EXPECTED_WALL} expected_cost=${EXPECTED_COST}"

# 1. (Re)convert v2 -> Dataset002 with test_holdout excluded (clean perimeter).
if [ ! -f "${nnUNet_raw}/${DS_TAG}/dataset.json" ]; then
    echo "[$(date -u +%FT%TZ)] converting v2 -> ${DS_TAG} (excluding 25 test_holdout cases)"
    /opt/pytorch/bin/python scripts/convert_v2_to_nnunet.py \
        --out-dir "$nnUNet_raw" \
        --dataset-id "$DATASET_ID" \
        --dataset-name "$DATASET_NAME" \
        --test-holdout-csv data/processed/audit_v2_corrected/test_holdout.csv
fi

# 2. plan_and_preprocess (idempotent if already done).
if [ ! -f "${nnUNet_preprocessed}/${DS_TAG}/nnUNetPlans.json" ]; then
    echo "[$(date -u +%FT%TZ)] nnU-Net plan + preprocess on ${DS_TAG}"
    /opt/pytorch/bin/nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 2d --verify_dataset_integrity
fi

# 3. Train all 5 folds sequentially.
for fold in 0 1 2 3 4; do
    echo "[$(date -u +%FT%TZ)] === fold ${fold} ==="
    /opt/pytorch/bin/nnUNetv2_train "$DATASET_ID" 2d "$fold" -tr "$TRAINER"
done

# 4. find_best_configuration emits a cross-fold mean Dice.
echo "[$(date -u +%FT%TZ)] === find_best_configuration ==="
/opt/pytorch/bin/nnUNetv2_find_best_configuration "$DATASET_ID" -c 2d -tr "$TRAINER" || true

# 5. Synthesize a sentinel JSON.
SENTINEL="experiments/results/nnunet_2d_5fold_${TRAINER}.json"
mkdir -p experiments/results
/opt/pytorch/bin/python -c "
import json, os, sys
from pathlib import Path
results_root = Path(os.environ['nnUNet_results']) / '${DS_TAG}' / '${TRAINER}__nnUNetPlans__2d'
folds = []
for fd in sorted(results_root.glob('fold_*')):
    summary = fd / 'progress.png'  # nnU-Net keeps progress.png + checkpoint_final.pth + final_summary.json
    final_json = fd / 'validation' / 'summary.json'
    rec = {'fold': int(fd.name.split('_')[1]), 'fold_dir': str(fd)}
    if final_json.exists():
        s = json.loads(final_json.read_text())
        rec['mean_dice'] = float(s.get('mean', {}).get('Dice', float('nan')))
    folds.append(rec)
import numpy as np
dices = [f.get('mean_dice', float('nan')) for f in folds if not (f.get('mean_dice') is None or f['mean_dice'] != f['mean_dice'])]
out = {
    'phase': 'nnunet_2d_5fold_${TRAINER}',
    'trainer': '${TRAINER}',
    'dataset': '${DS_TAG}',
    'mean_dice': float(np.mean(dices)) if dices else None,
    'std_dice': float(np.std(dices, ddof=0)) if dices else None,
    'n_folds': len(dices),
    'folds': folds,
    'phase1_2_5fold_mean_reference': 0.6946,
    'user_target_dice': 0.75,
    'thesis_target_dice': 0.78,
}
Path('${SENTINEL}').write_text(json.dumps(out, indent=2))
print('wrote ${SENTINEL}:', out.get('mean_dice'), '+/-', out.get('std_dice'))
"

# 6. dvc-add + push the per-fold results dirs (per feedback_save_model_weights).
echo "[$(date -u +%FT%TZ)] === dvc add + push ==="
mkdir -p ai/models/checkpoints/nnunet
RESULTS_DIR="ai/models/checkpoints/nnunet/${TRAINER}_${DS_TAG}"
cp -r "${nnUNet_results}/${DS_TAG}/${TRAINER}__nnUNetPlans__2d" "$RESULTS_DIR" 2>/dev/null || true
/opt/pytorch/bin/dvc add "$RESULTS_DIR" 2>&1 | tail -3 || true
/opt/pytorch/bin/dvc push "${RESULTS_DIR}.dvc" 2>&1 | tail -3 || true
git add -f "${RESULTS_DIR}.dvc" "$SENTINEL" 2>/dev/null || true
git -c user.email=ec2-auto@scoliosis -c user.name="EC2 auto" commit -m "nnunet 2d 5fold: $TRAINER on $DS_TAG" 2>&1 | tail -3 || true
git push origin HEAD 2>&1 | tail -3 || echo "git push failed (no GitHub creds on EC2 — expected)"

# 7. shutdown.
echo "[$(date -u +%FT%TZ)] === shutdown -h +2 ==="
sudo shutdown -h +2 "nnunet 2d 5fold complete; auto-stop"
