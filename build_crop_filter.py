from pathlib import Path
import random
import shutil

random.seed(42)

RICE_DIR = Path("dataset/rice_clean")
EGGPLANT_DIR = Path("dataset/eggplant_clean")
OTHER_DIR = Path("dataset/other_human")

OUTPUT_DIR = Path("dataset/crop_filter")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(base_dir, use_splits=True):
    images = []

    if use_splits:
        for split in ["train", "val", "test"]:
            split_dir = base_dir / split
            if split_dir.exists():
                for file in split_dir.rglob("*"):
                    if file.suffix.lower() in EXTS:
                        images.append(file)
    else:
        for file in base_dir.rglob("*"):
            if file.suffix.lower() in EXTS:
                images.append(file)

    return images


rice_images = collect_images(RICE_DIR, use_splits=True)
eggplant_images = collect_images(EGGPLANT_DIR, use_splits=True)
other_images = collect_images(OTHER_DIR, use_splits=False)

print("Rice:", len(rice_images))
print("Eggplant:", len(eggplant_images))
print("Other:", len(other_images))

min_count = min(len(rice_images), len(eggplant_images), len(other_images))
print("Using per class:", min_count)

rice_images = random.sample(rice_images, min_count)
eggplant_images = random.sample(eggplant_images, min_count)
other_images = random.sample(other_images, min_count)

data = {
    "rice": rice_images,
    "eggplant": eggplant_images,
    "other": other_images
}

if OUTPUT_DIR.exists():
    raise Exception(f"{OUTPUT_DIR} already exists. Delete it first.")

for class_name, files in data.items():
    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, split_files in split_map.items():
        dest_dir = OUTPUT_DIR / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(split_files):
            new_name = f"{class_name}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dest_dir / new_name)

    print(
        f"{class_name}: "
        f"train={len(train_files)}, "
        f"val={len(val_files)}, "
        f"test={len(test_files)}"
    )

print("\nDone:", OUTPUT_DIR)