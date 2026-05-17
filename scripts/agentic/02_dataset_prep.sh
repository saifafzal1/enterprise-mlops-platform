#!/usr/bin/env bash
# Mac A — Dataset preparation
# Generates synthetic dataset OR ingests real Jira pairs from data/agentic/raw/
# TOKEN-SAVING: all dataset logic lives in src/agentic/dataset_builder.py
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/02_dataset_prep.log"
DATA="$PROJECT/data/agentic"
mkdir -p "$DATA/raw" "$DATA/processed" "$PROJECT/logs"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export MLFLOW_TRACKING_URI="http://localhost:5000"

echo "=== [02] Dataset Preparation ===" | tee "$LOG"

# ── Build dataset (synthetic if no real data present) ────────────────────────
python3 "$PROJECT/src/agentic/dataset_builder.py" \
  --cypress-out  "$DATA/processed/cypress_train.jsonl" \
  --playwright-out "$DATA/processed/playwright_train.jsonl" \
  --cypress-test  "$DATA/processed/cypress_test.jsonl" \
  --playwright-test "$DATA/processed/playwright_test.jsonl" \
  --real-cypress  "$DATA/raw/cypress_pairs.jsonl" \
  --real-playwright "$DATA/raw/playwright_pairs.jsonl" \
  --n-synthetic 280 \
  --test-split 0.15 \
  2>&1 | tee -a "$LOG"

# ── DVC track datasets ────────────────────────────────────────────────────────
cd "$PROJECT"
dvc add data/agentic/processed/ 2>&1 | tee -a "$LOG" || echo "DVC add skipped" | tee -a "$LOG"

# ── Print counts ──────────────────────────────────────────────────────────────
python3 -c "
import jsonlines, os
for f in ['cypress_train','playwright_train','cypress_test','playwright_test']:
    path = 'data/agentic/processed/' + f + '.jsonl'
    if os.path.exists(path):
        with jsonlines.open(path) as r:
            n = sum(1 for _ in r)
        print(f'{f}: {n} pairs')
" | tee -a "$LOG"

echo "=== Dataset prep complete ===" | tee -a "$LOG"
