#!/bin/bash
# RUN ON: Mac A
# PURPOSE: Download datasets, DVC-track them, push to Mac B remote, push code to GitHub
set -euo pipefail

PROJECT="$HOME/Documents/enterprise-mlops-platform"
cd "$PROJECT"
source .venv/bin/activate

echo "============================================"
echo " STEP 1/5 — Downloading Cats vs Dogs"
echo "============================================"
mkdir -p data/raw/dogs-vs-cats
ZIP="data/raw/cats_and_dogs.zip"
curl -L "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip" -o "$ZIP"
unzip -q "$ZIP" -d data/raw/
mv data/raw/cats_and_dogs_filtered/* data/raw/dogs-vs-cats/
rm -rf data/raw/cats_and_dogs_filtered "$ZIP"
echo "  Done. Train: $(find data/raw/dogs-vs-cats/train -name '*.jpg' | wc -l | tr -d ' ') images"

echo "============================================"
echo " STEP 2/5 — Downloading Heart Disease CSV"
echo "============================================"
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" \
  -o data/raw/heart_disease.csv
echo "  Done. Rows: $(wc -l < data/raw/heart_disease.csv | tr -d ' ')"

echo "============================================"
echo " STEP 3/5 — Tracking datasets with DVC"
echo "============================================"
dvc add data/raw/dogs-vs-cats data/raw/heart_disease.csv
git add data/raw/.gitignore data/raw/dogs-vs-cats.dvc data/raw/heart_disease.csv.dvc
git commit -m "data: add raw datasets via DVC"
echo "  Committed DVC metadata"

echo "============================================"
echo " STEP 4/5 — Pushing data to Mac B (DVC remote)"
echo "============================================"
dvc push
echo "  DVC push complete"

echo "============================================"
echo " STEP 5/5 — Pushing code to GitHub"
echo "============================================"
git push origin main
echo "  GitHub push complete"

echo ""
echo "✓ Data setup complete. Next: bash scripts/02_train.sh"
