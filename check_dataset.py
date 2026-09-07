from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Class Mapping:", train_dataset.class_to_idx)

print("Train Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Test Images:", len(test_dataset))

from collections import Counter

train_counts = Counter(train_dataset.targets)

print("\nTrain class distribution:")

for class_index, count in sorted(train_counts.items()):
    print(f"{train_dataset.classes[class_index]}: {count}")
    import torch

    class_counts = torch.tensor(
        [train_counts[i] for i in range(len(train_dataset.classes))],
        dtype=torch.float
    )

    class_weights = len(train_dataset) / (
            len(train_dataset.classes) * class_counts
    )

    print("\nClass Weights:")

    for i, weight in enumerate(class_weights):
        print(
            f"{train_dataset.classes[i]}: {weight:.4f}"
        )