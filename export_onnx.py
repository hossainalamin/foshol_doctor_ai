import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "models/mobilenetv3_finetuned_best.pth"
ONNX_PATH = "models/mobilenetv3_rice.onnx"

DEVICE = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

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

model.eval()

dummy_input = torch.randn(
    1, 3, 224, 224
)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=17
)

print("ONNX model saved at:", ONNX_PATH)