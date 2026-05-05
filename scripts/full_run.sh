#!/usr/bin/env bash
# Full pipeline runner — chains every training job sequentially after the
# fidelity run completes. Each step writes its own log under logs/.
# Each `run` call is idempotent thanks to cfg-hash caching, so re-running
# just picks up where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."

# Repo root must be on PYTHONPATH so `import ai` works from any cwd.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs experiments/results

# Wait for any in-flight fidelity / training run to finish.
# Detect by process name pattern (works whether or not run_a.pid exists).
echo "[full_run] waiting on any in-flight env/bin/python training process"
while true; do
    if ! pgrep -f "env/bin/python.*ai.training.trainer\|env/bin/python -c.*from ai.training.trainer" > /dev/null 2>&1; then
        break
    fi
    sleep 30
done
echo "[full_run] no training process detected — proceeding"

echo "[full_run] starting Phase 0 ablations"
env/bin/python scripts/phase0_ablations.py 2>&1 | tee logs/phase0_ablations.log

echo "[full_run] starting Phase 1.1 (TXRV) single-split run"
env/bin/python -c "
import yaml, copy, logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from ai.training.trainer import run

with open('params.yaml') as f:
    p = yaml.safe_load(f)

# Read the Phase 0 winners
with open('experiments/results/phase0_summary.json') as f:
    ph0 = json.load(f)
best = max(ph0['results'], key=lambda r: r['best_val_dice'])
print(f\"Phase 0 winner: {best['name']} dice={best['best_val_dice']:.3f}\")

p['train']['encoder_name'] = 'txrv-resnet50'
p['train']['preprocess']['clahe_mode'] = best['clahe_mode']
p['train']['loss']['boundary_lambda'] = best['boundary_lambda']
p['train']['preprocess']['normalization'] = 'div255'  # txrv normalization is built into the encoder
p['train']['ema']['enabled'] = True
p['train']['batch_size'] = 2  # bigger encoder, halve batch
p['train']['lr_dec'] = 5.0e-4   # halve since batch halved

result = run(p, use_cache=False)
print('TXRV RUN:', result)
with open('experiments/results/phase1_1_txrv.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
" 2>&1 | tee logs/phase1_1_txrv.log

echo "[full_run] starting Phase 1.2 (TXRV + ROI mask crop)"
env/bin/python -c "
import yaml, copy, logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from ai.training.trainer import run

with open('params.yaml') as f:
    p = yaml.safe_load(f)
with open('experiments/results/phase0_summary.json') as f:
    ph0 = json.load(f)
best = max(ph0['results'], key=lambda r: r['best_val_dice'])

p['train']['encoder_name'] = 'txrv-resnet50'
p['train']['preprocess']['clahe_mode'] = best['clahe_mode']
p['train']['preprocess']['roi_crop'] = 'from_mask'
p['train']['loss']['boundary_lambda'] = best['boundary_lambda']
p['train']['ema']['enabled'] = True
p['train']['batch_size'] = 2
p['train']['lr_dec'] = 5.0e-4

result = run(p, use_cache=False)
print('TXRV + ROI RUN:', result)
with open('experiments/results/phase1_2_txrv_roi.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
" 2>&1 | tee logs/phase1_2_txrv_roi.log

echo "[full_run] starting Phase 1 5-fold CV gate (TXRV + ROI)"
env/bin/python scripts/cv5_train.py --epochs 60 --out experiments/results/cv5_phase1.json 2>&1 | tee logs/cv5_phase1.log

echo "[full_run] DONE"
