from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("dataset/other_human")
OUTPUT.mkdir(parents=True, exist_ok=True)

LIMIT = 1000

print("Loading human image dataset...")

dataset = load_dataset(
    "marcelohaps/lfw",
    split="train",
    streaming=True
)

# Randomize streamed samples somewhat
dataset = dataset.shuffle(
    seed=42,
    buffer_size=5000
)

count = 0

for item in dataset:

    image = item["image"].convert("RGB")

    image.save(
        OUTPUT / f"human_{count:05d}.jpg",
        quality=90
    )

    count += 1

    if count % 100 == 0:
        print(f"Downloaded: {count}")

    if count >= LIMIT:
        break

print(f"\nDone. Human images: {count}")
print("Saved in:", OUTPUT)