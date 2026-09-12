import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

DATA_DIR = "dataset/crop_filter"
MODEL_PATH = "models/crop_filter_best.pth"

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_dataset.classes)
print("Device:", DEVICE)

counts = torch.bincount(torch.tensor(train_dataset.targets)).float()
weights = len(train_dataset) / (len(counts) * counts)

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(train_dataset.classes)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weights.to(DEVICE)
)

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

best_val_accuracy = 0.0

Path("models").mkdir(exist_ok=True)

for epoch in range(EPOCHS):

    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train: {train_acc:.2f}% | "
        f"Val: {val_acc:.2f}%"
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes
        }, MODEL_PATH)

        print("Best model saved")

print("\nTraining completed")
print("Best Validation Accuracy:", best_val_accuracy)
print("Saved:", MODEL_PATH)