#!/usr/bin/env bash
# Mac A — QLoRA fine-tuning of Gemma 4 E4B (Cypress + Playwright)
# TOKEN-SAVING: all training logic in src/agentic/trainer.py
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/05_finetune_gemma4.log"
DATA="$PROJECT/data/agentic/processed"
MODELS="$PROJECT/models/agentic"
mkdir -p "$MODELS" "$PROJECT/logs"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Try Gemma 4 E4B; fall back to Gemma 3 4B if not yet on HuggingFace
MODEL_ID="google/gemma-3-4b-it"
python3 -c "
from huggingface_hub import model_info
try:
    model_info('google/gemma-4-4b-it')
    print('google/gemma-4-4b-it')
except:
    print('google/gemma-3-4b-it')
" > /tmp/gemma_model_id.txt 2>/dev/null || true
RESOLVED=$(cat /tmp/gemma_model_id.txt 2>/dev/null || echo "$MODEL_ID")
echo "Using model: $RESOLVED" | tee "$LOG"

EPOCHS=3
LR="2e-4"
BATCH=2    # Gemma 4B needs smaller batch on 48GB
MAX_LEN=1024

echo "=== [05] Fine-tuning Gemma 4 E4B ===" | tee -a "$LOG"

for FRAMEWORK in cypress playwright; do
  echo "--- Fine-tuning Gemma4 × $FRAMEWORK ---" | tee -a "$LOG"
  python3 "$PROJECT/src/agentic/trainer.py" \
    --model-id    "$RESOLVED" \
    --framework   "$FRAMEWORK" \
    --train-data  "$DATA/${FRAMEWORK}_train.jsonl" \
    --test-data   "$DATA/${FRAMEWORK}_test.jsonl" \
    --output-dir  "$MODELS/gemma4_${FRAMEWORK}" \
    --epochs      "$EPOCHS" \
    --lr          "$LR" \
    --batch-size  "$BATCH" \
    --max-length  "$MAX_LEN" \
    --mlflow-experiment "gemma4_finetune" \
    --run-name    "gemma4_${FRAMEWORK}" \
    2>&1 | tee -a "$LOG"
  echo "Saved to $MODELS/gemma4_${FRAMEWORK}" | tee -a "$LOG"
done

echo "=== Gemma 4 fine-tuning complete ===" | tee -a "$LOG"
