#!/usr/bin/env bash
# Mac A — QLoRA fine-tuning of Phi-3-mini (Cypress + Playwright)
# Uses MLX-LoRA (Apple Silicon native QLoRA equivalent)
# TOKEN-SAVING: all training logic in src/agentic/trainer.py
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/04_finetune_phi3.log"
DATA="$PROJECT/data/agentic/processed"
MODELS="$PROJECT/models/agentic"
mkdir -p "$MODELS" "$PROJECT/logs"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export MLFLOW_TRACKING_URI="http://localhost:5000"

MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
EPOCHS=3
LR="2e-4"
BATCH=4
MAX_LEN=1024

echo "=== [04] Fine-tuning Phi-3-mini ===" | tee "$LOG"
echo "Model: $MODEL_ID | Epochs: $EPOCHS | LR: $LR" | tee -a "$LOG"

for FRAMEWORK in cypress playwright; do
  echo "--- Fine-tuning Phi-3-mini × $FRAMEWORK ---" | tee -a "$LOG"
  python3 "$PROJECT/src/agentic/trainer.py" \
    --model-id    "$MODEL_ID" \
    --framework   "$FRAMEWORK" \
    --train-data  "$DATA/${FRAMEWORK}_train.jsonl" \
    --test-data   "$DATA/${FRAMEWORK}_test.jsonl" \
    --output-dir  "$MODELS/phi3_${FRAMEWORK}" \
    --epochs      "$EPOCHS" \
    --lr          "$LR" \
    --batch-size  "$BATCH" \
    --max-length  "$MAX_LEN" \
    --mlflow-experiment "phi3_finetune" \
    --run-name    "phi3_${FRAMEWORK}" \
    2>&1 | tee -a "$LOG"
  echo "Saved to $MODELS/phi3_${FRAMEWORK}" | tee -a "$LOG"
done

echo "=== Phi-3-mini fine-tuning complete ===" | tee -a "$LOG"
echo "Check MLflow at http://localhost:5000 for loss curves and F1 scores" | tee -a "$LOG"
