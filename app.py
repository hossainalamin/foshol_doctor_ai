from io import BytesIO

import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import models, transforms
import joblib
from pydantic import BaseModel


# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="Fasol Doctor AI API",
    version="1.0"
)

# =====================================================
# Request Models
# =====================================================

class TextPredictionRequest(BaseModel):
    text: str

# =====================================================
# Configuration
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/mobilenetv3_finetuned_best.pth"
TEXT_MODEL_PATH = "models/bangla_text_model.joblib"
CONFIDENCE_THRESHOLD = 70.0


# =====================================================
# Load Image Model
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]

IMAGE_SIZE = checkpoint.get(
    "image_size",
    224
)


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


print("AI Model loaded successfully")
print("Device:", DEVICE)
print("Classes:", class_names)

# =====================================================
# Load Bangla Text Model
# =====================================================

text_model = joblib.load(
    TEXT_MODEL_PATH
)

print("Bangla text model loaded successfully")


# =====================================================
# Image Preprocessing
# =====================================================

transform = transforms.Compose([
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
# Health API
# =====================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "Fasol Doctor AI"
    }


# =====================================================
# Predict API
# =====================================================

@app.post("/predict/image")
async def predict(
    image: UploadFile = File(...)
):

    # Check file type
    if not image.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    try:

        contents = await image.read()

        pil_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read image."
        )


    # Preprocess
    input_tensor = transform(
        pil_image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )


    # Prediction
    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    disease = class_names[
        predicted.item()
    ]

    confidence_value = (
        confidence.item()
        * 100
    )


    needs_expert_review = (
        confidence_value
        < CONFIDENCE_THRESHOLD
    )


    return {
        "disease": disease,
        "confidence": round(
            confidence_value,
            2
        ),
        "needsExpertReview": needs_expert_review
    }
@app.post("/predict/text")
def predict_text(request: TextPredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    probabilities = text_model.predict_proba([text])[0]
    classes = text_model.classes_

    best_index = int(probabilities.argmax())

    disease = str(classes[best_index])

    confidence = float(
        probabilities[best_index] * 100
    )

    needs_expert_review = bool(
        confidence < CONFIDENCE_THRESHOLD
    )

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "needsExpertReview": needs_expert_review
    }