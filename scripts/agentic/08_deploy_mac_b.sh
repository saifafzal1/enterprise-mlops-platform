#!/usr/bin/env bash
# Mac A → Mac B — Sync fine-tuned models and redeploy FastAPI
# TOKEN-SAVING: model sync via rsync + Docker rebuild on Mac B
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
LOG="$PROJECT/logs/08_deploy_mac_b.log"
REMOTE="mac-b:~/Documents/enterprise-mlops-platform-sync"
mkdir -p "$PROJECT/logs"

echo "=== [08] Deploy Agentic Models to Mac B ===" | tee "$LOG"

# ── Sync fine-tuned model adapters (exclude full weights, only adapters) ──────
echo "Syncing model adapters to Mac B..." | tee -a "$LOG"
rsync -av --progress \
  --include="*/" \
  --include="adapter_*.safetensors" \
  --include="adapter_config.json" \
  --include="tokenizer*" \
  --include="special_tokens_map.json" \
  --exclude="*" \
  "$PROJECT/models/agentic/" \
  "$REMOTE/models/agentic/" \
  2>&1 | tee -a "$LOG"

# ── Sync source code ──────────────────────────────────────────────────────────
rsync -av \
  "$PROJECT/src/agentic/" \
  "$REMOTE/src/agentic/" \
  2>&1 | tee -a "$LOG"

rsync -av \
  "$PROJECT/src/api/main.py" \
  "$REMOTE/src/api/main.py" \
  2>&1 | tee -a "$LOG"

# ── Rebuild and restart API on Mac B ─────────────────────────────────────────
echo "Rebuilding Docker image on Mac B..." | tee -a "$LOG"
ssh mac-b "
  export PATH=/usr/local/bin:\$PATH
  cd ~/Documents/enterprise-mlops-platform-sync
  docker compose build mlops-api 2>&1 | tail -5
  docker compose up -d mlops-api
  sleep 10
  curl -s http://localhost:8000/health
" 2>&1 | tee -a "$LOG"

echo "=== Deployment complete ===" | tee -a "$LOG"
echo "API: http://192.168.1.178:8000/docs" | tee -a "$LOG"
