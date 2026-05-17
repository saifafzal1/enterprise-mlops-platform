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

SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
}

def main():
    for raw_split, out_split in SPLIT_MAP.items():
        for cls in ("cats", "dogs"):
            src = RAW_DIR / raw_split / cls
            dst = OUT_DIR / out_split / cls
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            n = len(list(dst.glob("*.jpg")))
            print(f"{out_split}/{cls}: {n} images")

if __name__ == "__main__":
    main()
