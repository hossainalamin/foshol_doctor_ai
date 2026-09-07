import os
import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =====================================================
# Configuration
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

OLD_MODEL_PATH = "models/mobilenetv3_best.pth"
NEW_MODEL_PATH = "models/mobilenetv3_finetuned_best.pth"

os.makedirs("models", exist_ok=True)


# =====================================================
# Better Training Augmentation
# =====================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=25
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation remains clean
val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# Load Dataset
# =====================================================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)

class_names = train_dataset.classes


print("Device:", DEVICE)
print("Classes:", class_names)

print(
    "Train Images:",
    len(train_dataset)
)

print(
    "Validation Images:",
    len(val_dataset)
)


# =====================================================
# Calculate Class Weights
# =====================================================

train_counts = Counter(
    train_dataset.targets
)

class_counts = torch.tensor(
    [
        train_counts[i]
        for i in range(len(class_names))
    ],
    dtype=torch.float
)

class_weights = (
    len(train_dataset)
    /
    (
        len(class_names)
        * class_counts
    )
)

class_weights = class_weights.to(
    DEVICE
)


print("\nClass Weights:")

for i, weight in enumerate(class_weights):

    print(
        f"{class_names[i]}: "
        f"{weight.item():.4f}"
    )


# =====================================================
# DataLoader
# =====================================================

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


print(
    "\nTrain batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# =====================================================
# Rebuild MobileNetV3
# =====================================================

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)


# =====================================================
# Load Previous Best Model
# =====================================================

checkpoint = torch.load(
    OLD_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "\nPrevious Best Validation Accuracy:",
    checkpoint["best_val_accuracy"]
)


# =====================================================
# Freeze Entire Feature Extractor First
# =====================================================

for param in model.features.parameters():
    param.requires_grad = False


# =====================================================
# Unfreeze Last 3 Feature Blocks
# =====================================================

for block in model.features[-3:]:

    for param in block.parameters():
        param.requires_grad = True


# Classifier stays trainable
for param in model.classifier.parameters():
    param.requires_grad = True


model = model.to(DEVICE)


# =====================================================
# Show Trainable Layers
# =====================================================

print("\nTrainable Layers:")

for name, param in model.named_parameters():

    if param.requires_grad:
        print(name)


# =====================================================
# Weighted Loss
# =====================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =====================================================
# Differential Learning Rates
# =====================================================
# Feature layers:
# very small learning rate
#
# Classifier:
# slightly larger learning rate
# =====================================================

optimizer = optim.Adam(
    [
        {
            "params": model.features[-3:].parameters(),
            "lr": 0.00001
        },

        {
            "params": model.classifier.parameters(),
            "lr": 0.0001
        }
    ]
)


# =====================================================
# Learning Rate Scheduler
# =====================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# =====================================================
# Fine-tuning Setup
# =====================================================

best_val_accuracy = checkpoint[
    "best_val_accuracy"
]

best_model_weights = copy.deepcopy(
    model.state_dict()
)


print(
    "\nStarting fine-tuning..."
)


# =====================================================
# Fine-tuning Loop
# =====================================================

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 55)


    # =================================================
    # TRAIN
    # =================================================

    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0


    for batch_number, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        train_loss_total += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += (
            labels.size(0)
        )


        train_correct += (
            predicted == labels
        ).sum().item()


        if (
            batch_number + 1
        ) % 100 == 0:

            print(
                f"Processed "
                f"{batch_number + 1}"
                f"/{len(train_loader)} "
                f"train batches"
            )


    train_loss = (
        train_loss_total
        / train_total
    )


    train_accuracy = (
        100
        * train_correct
        / train_total
    )


    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss_total
        / val_total
    )


    val_accuracy = (
        100
        * val_correct
        / val_total
    )


    # =================================================
    # Scheduler
    # =================================================

    scheduler.step(
        val_accuracy
    )


    feature_lr = (
        optimizer.param_groups[0]["lr"]
    )

    classifier_lr = (
        optimizer.param_groups[1]["lr"]
    )


    # =================================================
    # Show Results
    # =================================================

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Feature LR: {feature_lr:.7f}"
    )

    print(
        f"Classifier LR: "
        f"{classifier_lr:.7f}"
    )


    # =================================================
    # Save Best Fine-tuned Model
    # =================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy
            },

            NEW_MODEL_PATH
        )


        print(
            "✓ New best fine-tuned model saved!"
        )


# =====================================================
# Finished
# =====================================================

model.load_state_dict(
    best_model_weights
)


print(
    "\nFine-tuning completed."
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Fine-tuned model:",
    NEW_MODEL_PATH
)