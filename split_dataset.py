"""
split_dataset.py
----------------
Splits images from dataset/raw/ into dataset/train/, dataset/val/, dataset/test/
using a stratified split: 10 train / 2 val / 3 test per class.

Copies (does not move) images so raw/ remains intact.

Run from the project root:
    python split_dataset.py
"""

import os
import shutil
import random

SEED = 42
RAW_DIR = os.path.join("dataset", "raw")
SPLITS = {
    "train": 10,
    "val":   2,
    "test":  3,
}


def get_class_folders(raw_dir):
    return sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])


def get_images(class_path):
    return sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith(".jpg") and not f.startswith(".")
    ])


def split_class(class_folder, raw_dir, splits, seed):
    class_path = os.path.join(raw_dir, class_folder)
    images = get_images(class_path)

    total_needed = sum(splits.values())
    if len(images) != total_needed:
        raise ValueError(
            f"{class_folder}: expected {total_needed} images, found {len(images)}"
        )

    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)

    assigned = {}
    idx = 0
    for split, count in splits.items():
        assigned[split] = shuffled[idx: idx + count]
        idx += count

    return assigned


def copy_images(class_folder, raw_dir, assigned):
    counts = {}
    for split, files in assigned.items():
        dst_dir = os.path.join("dataset", split, class_folder)
        for f in files:
            src = os.path.join(raw_dir, class_folder, f)
            dst = os.path.join(dst_dir, f)
            shutil.copy2(src, dst)
        counts[split] = len(files)
    return counts


def main():
    class_folders = get_class_folders(RAW_DIR)
    print(f"Found {len(class_folders)} classes — splitting {SPLITS}\n")

    for class_folder in class_folders:
        assigned = split_class(class_folder, RAW_DIR, SPLITS, SEED)
        counts = copy_images(class_folder, RAW_DIR, assigned)
        print(f"  {class_folder}: "
              f"train={counts['train']}  val={counts['val']}  test={counts['test']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
