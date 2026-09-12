from pathlib import Path
import random
import shutil

SOURCE = Path("dataset/eggplant_dataset")
OUTPUT = Path("dataset/eggplant_clean")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

if OUTPUT.exists():
    raise Exception(
        f"{OUTPUT} already exists. Delete/rename it before running again."
    )

classes = sorted([
    folder.name
    for folder in (SOURCE / "train").iterdir()
    if folder.is_dir()
])

for class_name in classes:

    # collect images from old train + val + test
    images = []

    for split in ["train", "val", "test"]:
        folder = SOURCE / split / class_name

        for file in folder.iterdir():
            if file.suffix.lower() in extensions:
                images.append(file)

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split, files in split_data.items():

        destination = OUTPUT / split / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, source_file in enumerate(files):

            # unique filename prevents overwrite
            new_name = f"{class_name.replace(' ', '_')}_{index:05d}{source_file.suffix.lower()}"

            shutil.copy2(
                source_file,
                destination / new_name
            )

    print(
        f"{class_name}: "
        f"Total={total}, "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\n✅ Resplit completed.")
print("New dataset:", OUTPUT)