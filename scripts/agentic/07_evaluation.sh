#!/usr/bin/env bash
# Mac A — Full evaluation matrix: all models × both frameworks
# Produces the 12-cell F1 table + error taxonomy + ANOVA results
# TOKEN-SAVING: all eval logic in src/agentic/evaluator.py
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/07_evaluation.log"
DATA="$PROJECT/data/agentic/processed"
MODELS="$PROJECT/models/agentic"
RESULTS="$PROJECT/data/agentic/results"
mkdir -p "$RESULTS/full_eval" "$PROJECT/logs"

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export MLFLOW_TRACKING_URI="http://localhost:5000"

if [ -f "$PROJECT/.env" ]; then
  export $(grep -v '^#' "$PROJECT/.env" | xargs)
fi

echo "=== [07] Full Evaluation Matrix ===" | tee "$LOG"

# ── Fine-tuned model evaluation ───────────────────────────────────────────────
for MODEL_KEY in phi3 gemma4; do
  for FRAMEWORK in cypress playwright; do
    MODEL_DIR="$MODELS/${MODEL_KEY}_${FRAMEWORK}"
    if [ -d "$MODEL_DIR" ]; then
      echo "Evaluating $MODEL_KEY × $FRAMEWORK..." | tee -a "$LOG"
      python3 "$PROJECT/src/agentic/evaluator.py" \
        --model-dir  "$MODEL_DIR" \
        --framework  "$FRAMEWORK" \
        --test-data  "$DATA/${FRAMEWORK}_test.jsonl" \
        --output     "$RESULTS/full_eval/${MODEL_KEY}_${FRAMEWORK}.json" \
        --with-bmad \
        --bmad-url   "http://192.168.1.178:8001" \
        --mlflow-experiment "full_evaluation" \
        --run-name   "${MODEL_KEY}_${FRAMEWORK}_eval" \
        2>&1 | tee -a "$LOG"
    else
      echo "SKIP: $MODEL_DIR not found (not yet trained)" | tee -a "$LOG"
    fi
  done
done

# ── Comparison table + statistical tests ─────────────────────────────────────
echo "--- Generating F1 comparison table ---" | tee -a "$LOG"
python3 "$PROJECT/src/agentic/evaluator.py" \
  --mode comparison-table \
  --results-dir "$RESULTS/full_eval" \
  --baselines-dir "$RESULTS/baselines" \
  --output "$RESULTS/f1_comparison_table.csv" \
  --stats-output "$RESULTS/anova_wilcoxon.json" \
  2>&1 | tee -a "$LOG"

echo "=== Evaluation complete ===" | tee -a "$LOG"
echo "F1 table: $RESULTS/f1_comparison_table.csv" | tee -a "$LOG"
echo "Stats:    $RESULTS/anova_wilcoxon.json" | tee -a "$LOG"
