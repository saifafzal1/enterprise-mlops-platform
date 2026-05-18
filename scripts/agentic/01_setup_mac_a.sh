#!/usr/bin/env bash
# Mac A — environment setup for SLM fine-tuning
# Run once before any other agentic script
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
VENV="$PROJECT/.venv_agentic"
LOG="$PROJECT/logs/01_setup_mac_a.log"
mkdir -p "$PROJECT/logs"

echo "=== [01] Mac A Agentic Setup ===" | tee "$LOG"

# ── Python virtual environment ──────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "Created venv at $VENV" | tee -a "$LOG"
fi
source "$VENV/bin/activate"

pip install --upgrade pip | tee -a "$LOG"
pip install -r "$PROJECT/requirements_agentic.txt" | tee -a "$LOG"

# ── Verify MPS (Apple Silicon GPU) ──────────────────────────────────────────
python3 -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:',     torch.backends.mps.is_built())
" | tee -a "$LOG"

python3 -c "import mlx.core; print('MLX version:', mlx.core.__version__)" | tee -a "$LOG"

# ── Pull base models to local cache (~/.cache/huggingface) ───────────────────
echo "Pulling Phi-3-mini tokenizer/config (not weights) for offline check..." | tee -a "$LOG"
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)
print('Phi-3-mini tokenizer OK')
" | tee -a "$LOG"

echo "Pulling Gemma-3-4b tokenizer..." | tee -a "$LOG"
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-3-4b-it', trust_remote_code=True)
print('Gemma-3-4b tokenizer OK')
" | tee -a "$LOG"

# ── MLflow server (background) ───────────────────────────────────────────────
if ! pgrep -f "mlflow server" > /dev/null; then
  source "$PROJECT/.venv/bin/activate" 2>/dev/null || true
  nohup mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$PROJECT/mlruns" \
    --default-artifact-root "$PROJECT/mlartifacts" \
    >> "$PROJECT/logs/mlflow.log" 2>&1 &
  echo "MLflow server started (PID $!)" | tee -a "$LOG"
else
  echo "MLflow server already running" | tee -a "$LOG"
fi

# ── Node.js check (needed for BMAD validator locally) ───────────────────────
if command -v node &> /dev/null; then
  echo "Node.js: $(node --version)" | tee -a "$LOG"
else
  echo "Node.js not found — install via: brew install node" | tee -a "$LOG"
fi

echo "=== Setup complete ===" | tee -a "$LOG"
