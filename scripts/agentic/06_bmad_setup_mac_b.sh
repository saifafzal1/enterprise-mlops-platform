#!/usr/bin/env bash
# Mac B — BMAD validation infrastructure setup
# Run this ON MAC B (ssh mac-b "bash -s" < scripts/agentic/06_bmad_setup_mac_b.sh)
# TOKEN-SAVING: all BMAD logic in docker/bmad/ and src/agentic/bmad_agent.py
set -euo pipefail

SYNC="$HOME/Documents/enterprise-mlops-platform-sync"
LOG="$SYNC/logs/06_bmad_setup.log"
mkdir -p "$SYNC/logs"

echo "=== [06] BMAD Validator Setup on Mac B ===" | tee "$LOG"

# ── Node.js ───────────────────────────────────────────────────────────────────
if ! command -v node &> /dev/null; then
  echo "Installing Node.js via brew..." | tee -a "$LOG"
  brew install node 2>&1 | tee -a "$LOG"
fi
echo "Node: $(node --version)" | tee -a "$LOG"

# ── Cypress + Playwright in a dedicated node project ─────────────────────────
BMAD_NODE="$SYNC/bmad_sandbox"
mkdir -p "$BMAD_NODE"
cd "$BMAD_NODE"

if [ ! -f "package.json" ]; then
  npm init -y 2>&1 | tee -a "$LOG"
fi

npm install --save-dev cypress@latest 2>&1 | tee -a "$LOG" || echo "Cypress install failed" | tee -a "$LOG"
npm install --save-dev @playwright/test@latest 2>&1 | tee -a "$LOG"
npx playwright install chromium 2>&1 | tee -a "$LOG"

echo "Cypress: $(npx cypress --version 2>/dev/null | head -1)" | tee -a "$LOG"
echo "Playwright: $(npx playwright --version 2>/dev/null)" | tee -a "$LOG"

# ── Build BMAD Docker container ───────────────────────────────────────────────
echo "Building BMAD validator container..." | tee -a "$LOG"
export PATH="/usr/local/bin:$PATH"
docker build -t bmad-validator:latest "$SYNC/docker/bmad/" 2>&1 | tee -a "$LOG"

# ── Start BMAD validator service ──────────────────────────────────────────────
docker rm -f bmad-validator 2>/dev/null || true
docker run -d \
  --name bmad-validator \
  -p 8001:8001 \
  --network enterprise-mlops-platform-sync_default \
  bmad-validator:latest 2>&1 | tee -a "$LOG"

sleep 5
curl -s http://localhost:8001/health && echo " BMAD validator healthy" | tee -a "$LOG"

echo "=== BMAD setup complete on Mac B ===" | tee -a "$LOG"
