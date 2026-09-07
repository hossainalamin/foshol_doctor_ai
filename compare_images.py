import os
import random
import matplotlib.pyplot as plt
from PIL import Image


TRAIN_FOLDER = "dataset/train/Brown spot"
VAL_FOLDER = "dataset/val/Brown spot"
TEST_FOLDER = "dataset/test/Brown spot"

random.seed(42)


def get_images(folder, count=8):
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    return random.sample(
        files,
        min(count, len(files))
    )


train_images = get_images(TRAIN_FOLDER)
val_images = get_images(VAL_FOLDER)
test_images = get_images(TEST_FOLDER)


plt.figure(figsize=(16, 10))


# TRAIN
for i, filename in enumerate(train_images):

    image = Image.open(
        os.path.join(
            TRAIN_FOLDER,
            filename
        )
    ).convert("RGB")

    plt.subplot(3, 8, i + 1)

    plt.imshow(image)

    plt.title("TRAIN")

    plt.axis("off")


# VALIDATION
for i, filename in enumerate(val_images):

    image = Image.open(
        os.path.join(
            VAL_FOLDER,
            filename
        )
    ).convert("RGB")

    plt.subplot(3, 8, i + 9)

    plt.imshow(image)

    plt.title("VAL")

    plt.axis("off")


# TEST
for i, filename in enumerate(test_images):

    image = Image.open(
        os.path.join(
            TEST_FOLDER,
            filename
        )
    ).convert("RGB")

    plt.subplot(3, 8, i + 17)

    plt.imshow(image)

    plt.title("TEST")

    plt.axis("off")


plt.tight_layout()

plt.show()