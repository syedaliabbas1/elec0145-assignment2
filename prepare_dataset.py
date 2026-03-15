"""
prepare_dataset.py
------------------
Converts all non-JPG images in dataset/raw/ to JPG and renames every image
(including existing JPGs) to a consistent format:

    [class_id]_[class_name]_[NN].jpg   e.g. 01_BMW_M3_2023_01.jpg

Run from the project root:
    python prepare_dataset.py
"""

import os
from PIL import Image

RAW_DIR = os.path.join("dataset", "raw")


def get_class_folders(raw_dir):
    return sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])


def process_class(class_folder, raw_dir):
    class_path = os.path.join(raw_dir, class_folder)
    files = sorted([
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
        and not f.startswith(".")
    ])

    converted = 0
    renamed = 0

    for idx, filename in enumerate(files, start=1):
        src_path = os.path.join(class_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{class_folder}_{idx:02d}.jpg"
        dst_path = os.path.join(class_path, new_name)

        if ext in (".jpg", ".jpeg"):
            # Already JPG — rename only if name differs
            if filename != new_name:
                os.rename(src_path, dst_path)
                renamed += 1
        else:
            # Convert to JPG, then delete original
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, "JPEG", quality=95)
            os.remove(src_path)
            converted += 1

    return converted, renamed


def main():
    class_folders = get_class_folders(RAW_DIR)
    print(f"Found {len(class_folders)} classes in {RAW_DIR}/\n")

    total_converted = 0
    total_renamed = 0

    for class_folder in class_folders:
        converted, renamed = process_class(class_folder, RAW_DIR)
        total_converted += converted
        total_renamed += renamed
        print(f"  {class_folder}: {converted} converted, {renamed} renamed")

    print(f"\nDone. {total_converted} files converted to JPG, "
          f"{total_renamed} existing JPGs renamed.")


if __name__ == "__main__":
    main()
