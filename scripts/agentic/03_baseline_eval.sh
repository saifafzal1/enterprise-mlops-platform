#!/usr/bin/env bash
# Mac A — Zero-shot LLM baseline evaluation
# Calls GPT-4o-mini, Claude Haiku, Gemini 1.5 Flash APIs
# TOKEN-SAVING: all eval logic lives in src/agentic/baseline_eval.py
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/03_baseline_eval.log"
DATA="$PROJECT/data/agentic/processed"
RESULTS="$PROJECT/data/agentic/results/baselines"
mkdir -p "$RESULTS" "$PROJECT/logs"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Expects API keys in .env file
if [ -f "$PROJECT/.env" ]; then
  export $(grep -v '^#' "$PROJECT/.env" | xargs)
fi

echo "=== [03] Baseline LLM Evaluation ===" | tee "$LOG"
echo "Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY in .env" | tee -a "$LOG"

for FRAMEWORK in cypress playwright; do
  echo "--- Framework: $FRAMEWORK ---" | tee -a "$LOG"
  python3 "$PROJECT/src/agentic/baseline_eval.py" \
    --framework "$FRAMEWORK" \
    --test-data  "$DATA/${FRAMEWORK}_test.jsonl" \
    --output-dir "$RESULTS/$FRAMEWORK" \
    --models "gpt-4o-mini,claude-haiku-4-5,phi3-mini-zeroshot,gemma4-zeroshot" \
    --mlflow-experiment "agentic_baselines" \
    2>&1 | tee -a "$LOG"
done

echo "=== Baseline eval complete — check MLflow at http://localhost:5000 ===" | tee -a "$LOG"
