import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================
# Configuration
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

MODEL_PATH = "models/mobilenetv3_finetuned_best.pth"

# =========================
# Test Transform
# =========================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Load Test Dataset
# =========================

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test Images:", len(test_dataset))
print("Test Classes:", test_dataset.classes)


# =========================
# Load Saved Checkpoint
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

print(
    "Best Validation Accuracy:",
    checkpoint["best_val_accuracy"]
)


# =========================
# Rebuild MobileNetV3
# =========================

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# =========================
# Test
# =========================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * correct / total
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)
# =========================
# Class-wise Test Accuracy
# =========================

class_correct = [0 for _ in class_names]
class_total = [0 for _ in class_names]

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        for label, prediction in zip(
            labels,
            predicted
        ):

            label_index = label.item()

            class_total[label_index] += 1

            if prediction.item() == label_index:
                class_correct[label_index] += 1


print("\nClass-wise Test Accuracy:")
from sklearn.metrics import confusion_matrix

all_true = []
all_pred = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_true.extend(
            labels.cpu().numpy()
        )

        all_pred.extend(
            predicted.cpu().numpy()
        )


cm = confusion_matrix(
    all_true,
    all_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nClass order:")
for i, name in enumerate(class_names):
    print(i, "=", name)

for i, class_name in enumerate(class_names):

    accuracy = (
        100
        * class_correct[i]
        / class_total[i]
    )

    print(
        f"{class_name}: "
        f"{accuracy:.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )
# =========================
# Validation Confusion Matrix
# =========================

from torchvision import datasets
from torch.utils.data import DataLoader

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=test_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_true = []
val_pred = []

model.eval()

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        val_true.extend(labels.cpu().numpy())
        val_pred.extend(predicted.cpu().numpy())


val_cm = confusion_matrix(
    val_true,
    val_pred
)

print("\nValidation Confusion Matrix:")
print(val_cm)