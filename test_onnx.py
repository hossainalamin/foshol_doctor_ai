import os
import numpy as np
import onnxruntime as ort

from PIL import Image
from torchvision import transforms


MODEL_PATH = "models/mobilenetv3_rice.onnx"

CLASS_NAMES = [
    "Blast",
    "Brown spot",
    "Healthy",
    "Leaf smut",
    "Rice Tungro",
    "Sheath blight"
]

TEST_IMAGE = "test_leaf.jpg"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = Image.open(
    TEST_IMAGE
).convert("RGB")

tensor = transform(image)

input_data = (
    tensor
    .unsqueeze(0)
    .numpy()
    .astype(np.float32)
)


session = ort.InferenceSession(
    MODEL_PATH
)

input_name = session.get_inputs()[0].name

outputs = session.run(
    None,
    {
        input_name: input_data
    }
)

logits = outputs[0][0]


# Softmax
exp_values = np.exp(
    logits - np.max(logits)
)

probabilities = (
    exp_values / exp_values.sum()
)


best_index = int(
    np.argmax(probabilities)
)

disease = CLASS_NAMES[
    best_index
]

confidence = (
    float(probabilities[best_index])
    * 100
)


print("Disease:", disease)
print(
    f"Confidence: {confidence:.2f}%"
)