#!/bin/bash
# RUN ON: Mac A
# PURPOSE: Preprocess data, train both models, track with MLflow + DVC, sync to Mac B
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
cd "$PROJECT"
source .venv/bin/activate

echo "============================================"
echo " STEP 1/7 — Installing ML dependencies"
echo "============================================"
pip install torch torchvision scikit-learn mlflow pandas joblib pillow -q
echo "  Dependencies ready"

echo "============================================"
echo " STEP 2/7 — Starting MLflow tracking server"
echo "============================================"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
pkill -f "mlflow server" 2>/dev/null || true
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns &
MLFLOW_PID=$!
sleep 4
echo "  MLflow running (PID $MLFLOW_PID) → http://127.0.0.1:5000"

echo "============================================"
echo " STEP 3/7 — Preprocessing images"
echo "============================================"
python src/image_classification/preprocess.py
echo "  Preprocessing done"

echo "============================================"
echo " STEP 4/7 — Training Heart Disease model"
echo "============================================"
python src/heart_disease/train.py
echo "  Heart model saved → models/heart_disease.pkl"

echo "============================================"
echo " STEP 5/7 — Training Image Classification model"
echo "============================================"
python src/image_classification/train.py
echo "  Image model saved → models/cats_vs_dogs.pt"

echo "============================================"
echo " STEP 6/7 — DVC tracking trained models"
echo "============================================"
dvc add models/cats_vs_dogs.pt models/heart_disease.pkl
git add models/.gitignore models/cats_vs_dogs.pt.dvc models/heart_disease.pkl.dvc
git commit -m "model: add trained artifacts"
dvc push
git push origin main
echo "  Models pushed to DVC remote and GitHub"

echo "============================================"
echo " STEP 7/7 — Syncing project to Mac B"
echo "============================================"
rsync -av \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='data/raw/dogs-vs-cats' \
  --exclude='*.pyc' \
  "$PROJECT/" \
  mac-b:~/Documents/enterprise-mlops-platform-sync/
echo "  Sync complete"

echo ""
echo "✓ Training complete!"
echo "  MLflow UI  → http://127.0.0.1:5000  (PID $MLFLOW_PID)"
echo "  Next: SSH to Mac B and run: bash scripts/03_docker.sh"
echo "  Stop MLflow when done: kill $MLFLOW_PID"
