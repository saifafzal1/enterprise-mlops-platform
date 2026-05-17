"""
The Google-filtered dataset already has train/cats, train/dogs, validation/cats, validation/dogs.
This script copies them into the standard processed layout expected by train.py:
  data/processed/cats_dogs/train/cats|dogs
  data/processed/cats_dogs/val/cats|dogs
"""
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw/dogs-vs-cats")
OUT_DIR = Path("data/processed/cats_dogs")

def main():
    # Microsoft dataset is already split by 01_data_setup.sh — just symlink to processed/
    for split in ("train", "val"):
        for cls in ("cats", "dogs"):
            src = RAW_DIR / split / cls
            dst = OUT_DIR / split / cls
            if not src.exists():
                raise FileNotFoundError(f"Missing: {src} — run 01_data_setup.sh first")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            n = len(list(dst.glob("*.jpg")))
            print(f"{split}/{cls}: {n} images")

if __name__ == "__main__":
    main()
