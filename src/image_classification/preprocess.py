"""
Organise raw Kaggle cats-vs-dogs download into ImageFolder structure:
  data/processed/cats_dogs/train/cats/  train/dogs/
  data/processed/cats_dogs/val/cats/    val/dogs/
"""
import os, shutil, random
from pathlib import Path

RAW_DIR   = Path("data/raw/dogs-vs-cats/train")   # kaggle unzip lands here
OUT_DIR   = Path("data/processed/cats_dogs")
VAL_SPLIT = 0.15
SEED      = 42

def main():
    random.seed(SEED)
    for split in ("train", "val"):
        for cls in ("cats", "dogs"):
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # kaggle dataset names files cat.0.jpg / dog.0.jpg
    all_files = list(RAW_DIR.glob("*.jpg"))
    cats = [f for f in all_files if f.name.startswith("cat")]
    dogs = [f for f in all_files if f.name.startswith("dog")]

    for cls_files, cls_name in [(cats, "cats"), (dogs, "dogs")]:
        random.shuffle(cls_files)
        n_val = int(len(cls_files) * VAL_SPLIT)
        splits = {"val": cls_files[:n_val], "train": cls_files[n_val:]}
        for split, files in splits.items():
            for f in files:
                shutil.copy(f, OUT_DIR / split / cls_name / f.name)

    for split in ("train", "val"):
        for cls in ("cats", "dogs"):
            n = len(list((OUT_DIR / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}: {n} images")

if __name__ == "__main__":
    main()
