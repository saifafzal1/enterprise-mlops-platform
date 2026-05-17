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
python3 - <<'PYEOF'
import requests, zipfile, shutil, pathlib

# Microsoft-hosted original Cats vs Dogs dataset (no auth required)
url  = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
dest = pathlib.Path("data/raw/cats_and_dogs.zip")

print(f"  Downloading Microsoft Cats vs Dogs (~786MB) ...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="\r")
print(f"\n  Downloaded {dest.stat().st_size / 1e6:.1f} MB")

print("  Extracting...")
with zipfile.ZipFile(dest) as z:
    z.extractall("data/raw/")
dest.unlink()

# Organise into train/val split with cats/ dogs/ subdirs
import random, os
random.seed(42)
raw     = pathlib.Path("data/raw/PetImages")
out     = pathlib.Path("data/raw/dogs-vs-cats")
for split in ("train", "val"):
    for cls in ("cats", "dogs"):
        (out / split / cls).mkdir(parents=True, exist_ok=True)

for cls, src_dir in [("cats", "Cat"), ("dogs", "Dog")]:
    files = [f for f in (raw / src_dir).glob("*.jpg") if f.stat().st_size > 1000]
    random.shuffle(files)
    n_val = int(len(files) * 0.15)
    for i, f in enumerate(files):
        split = "val" if i < n_val else "train"
        shutil.copy(f, out / split / cls / f.name)

shutil.rmtree(raw)
for split in ("train","val"):
    for cls in ("cats","dogs"):
        n = len(list((out/split/cls).glob("*.jpg")))
        print(f"  {split}/{cls}: {n} images")
PYEOF
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
