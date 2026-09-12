from io import BytesIO
import os
import secrets

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms


# =====================================================
# Environment
# =====================================================

load_dotenv()


# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="Fasol Doctor AI API",
    version="1.1",
    description=(
        "AI API for rice and eggplant disease prediction "
        "using images and Bangla text."
    ),
)


# =====================================================
# Disease Bangla Mapping
# =====================================================

RICE_DISEASE_BN = {
    "Blast": "ব্লাস্ট",
    "Brown spot": "ব্রাউন স্পট",
    "Healthy": "সুস্থ",
    "Leaf smut": "লিফ স্মাট",
    "Rice Tungro": "রাইস টুংরো",
    "Sheath blight": "শীথ ব্লাইট",
}

EGGPLANT_DISEASE_BN = {
    "Healthy Leaf": "সুস্থ পাতা",
    "Insect Pest Disease": "পোকামাকড়ের আক্রমণ",
    "Leaf Spot Disease": "পাতার দাগ রোগ",
    "Mosaic Virus Disease": "মোজাইক ভাইরাস রোগ",
    "Small Leaf Disease": "ছোট পাতা রোগ",
    "White Mold Disease": "হোয়াইট মোল্ড রোগ",
    "Wilt Disease": "ঢলে পড়া রোগ",
}


# =====================================================
# Request Models
# =====================================================

class TextPredictionRequest(BaseModel):
    text: str


# =====================================================
# Configuration
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RICE_IMAGE_MODEL_PATH = "models/mobilenetv3_finetuned_best.pth"
EGGPLANT_IMAGE_MODEL_PATH = "models/mobilenetv3_eggplant_best.pth"
RICE_TEXT_MODEL_PATH = "models/bangla_text_model.joblib"
EGGPLANT_TEXT_MODEL_PATH = "models/bangla_eggplant_text_model.joblib"

CONFIDENCE_THRESHOLD = 70.0
DEFAULT_IMAGE_SIZE = 224


# =====================================================
# API Key Configuration
# =====================================================

API_KEY = os.getenv("FASOL_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "FASOL_API_KEY is not configured. Please add it to the .env file."
    )

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="অবৈধ অথবা অনুপস্থিত API Key।",
        )

    return api_key


# =====================================================
# Image Quality Check
# =====================================================


def check_image_quality(pil_image: Image.Image):
    image = np.array(pil_image)

    height, width = image.shape[:2]

    if width < 224 or height < 224:
        return False, "ছবির resolution খুব কম। পরিষ্কার ছবি তুলুন।"

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    if blur_score < 80:
        return False, "ছবিটি অস্পষ্ট। আবার পরিষ্কার ছবি তুলুন।"

    brightness = gray.mean()

    if brightness < 45:
        return False, "ছবিটি অনেক অন্ধকার। আলোতে আবার ছবি তুলুন।"

    if brightness > 225:
        return False, "ছবিটি অতিরিক্ত উজ্জ্বল। আবার ছবি তুলুন।"

    return True, None


# =====================================================
# Model Loader
# =====================================================


def load_mobilenet_model(model_path: str):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    # Rice checkpoint uses "class_names".
    # Eggplant training checkpoint uses "classes".
    class_names = checkpoint.get("class_names")

    if class_names is None:
        class_names = checkpoint.get("classes")

    if class_names is None:
        raise RuntimeError(
            f"No class list found in checkpoint: {model_path}"
        )

    image_size = checkpoint.get(
        "image_size",
        DEFAULT_IMAGE_SIZE,
    )

    model = models.mobilenet_v3_small(
        weights=None,
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, list(class_names), image_size


# =====================================================
# Load Rice Image Model
# =====================================================

rice_image_model, rice_class_names, rice_image_size = load_mobilenet_model(
    RICE_IMAGE_MODEL_PATH
)

print("Rice image model loaded successfully")
print("Rice classes:", rice_class_names)


# =====================================================
# Load Eggplant Image Model
# =====================================================

eggplant_image_model, eggplant_class_names, eggplant_image_size = load_mobilenet_model(
    EGGPLANT_IMAGE_MODEL_PATH
)

print("Eggplant image model loaded successfully")
print("Eggplant classes:", eggplant_class_names)


# =====================================================
# Load Bangla Text Models
# =====================================================

rice_text_model = joblib.load(
    RICE_TEXT_MODEL_PATH
)

eggplant_text_model = joblib.load(
    EGGPLANT_TEXT_MODEL_PATH
)

print("Rice Bangla text model loaded successfully")
print("Eggplant Bangla text model loaded successfully")
print("Device:", DEVICE)


# =====================================================
# Image Preprocessing
# =====================================================


def build_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


rice_image_transform = build_image_transform(
    rice_image_size
)

eggplant_image_transform = build_image_transform(
    eggplant_image_size
)


# =====================================================
# Shared Image Helpers
# =====================================================


async def read_uploaded_image(image: UploadFile) -> Image.Image:
    if (
        image.content_type is None
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="অনুগ্রহ করে একটি সঠিক ছবি আপলোড করুন।",
        )

    try:
        contents = await image.read()
        pil_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="ছবিটি পড়া সম্ভব হয়নি।",
        ) from exc

    is_valid, error_message = check_image_quality(
        pil_image
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_message,
        )

    return pil_image


def predict_with_image_model(
    pil_image: Image.Image,
    model,
    image_transform,
    class_names,
    disease_mapping,
):
    input_tensor = (
        image_transform(pil_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    best_index = int(
        torch.argmax(probabilities).item()
    )

    disease_en = str(
        class_names[best_index]
    )

    disease_bn = disease_mapping.get(
        disease_en,
        disease_en,
    )

    confidence = float(
        probabilities[best_index].item()
        * 100
    )

    needs_expert_review = bool(
        confidence < CONFIDENCE_THRESHOLD
    )

    if needs_expert_review:
        message = (
            "রোগ শনাক্তকরণে AI যথেষ্ট নিশ্চিত নয়। "
            "বিশেষজ্ঞের পরামর্শ নিন।"
        )
    else:
        message = "রোগটি সফলভাবে শনাক্ত করা হয়েছে।"

    return {
        "disease": disease_bn,
        "confidence": round(confidence, 2),
        "needsExpertReview": needs_expert_review,
        "message": message,
    }


# =====================================================
# Health API
# =====================================================


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Fasol Doctor AI",
    }


# =====================================================
# Rice Image Prediction API
# Existing endpoint kept for app compatibility
# =====================================================


@app.post("/predict/image")
async def predict_rice_image(
    image: UploadFile = File(...),
    api_key: str = Security(verify_api_key),
):
    pil_image = await read_uploaded_image(image)

    return predict_with_image_model(
        pil_image=pil_image,
        model=rice_image_model,
        image_transform=rice_image_transform,
        class_names=rice_class_names,
        disease_mapping=RICE_DISEASE_BN,
    )


# =====================================================
# Eggplant Image Prediction API
# =====================================================


@app.post("/predict/eggplant/image")
async def predict_eggplant_image(
    image: UploadFile = File(...),
    api_key: str = Security(verify_api_key),
):
    pil_image = await read_uploaded_image(image)

    return predict_with_image_model(
        pil_image=pil_image,
        model=eggplant_image_model,
        image_transform=eggplant_image_transform,
        class_names=eggplant_class_names,
        disease_mapping=EGGPLANT_DISEASE_BN,
    )


# =====================================================
# Shared Text Prediction Helper
# =====================================================


def predict_with_text_model(
    text: str,
    model,
    disease_mapping,
):
    probabilities = model.predict_proba(
        [text]
    )[0]

    classes = model.classes_

    best_index = int(
        probabilities.argmax()
    )

    disease_en = str(
        classes[best_index]
    )

    disease_bn = disease_mapping.get(
        disease_en,
        disease_en,
    )

    confidence = float(
        probabilities[best_index]
        * 100
    )

    needs_expert_review = bool(
        confidence < CONFIDENCE_THRESHOLD
    )

    if needs_expert_review:
        message = (
            "রোগ শনাক্তকরণে AI যথেষ্ট নিশ্চিত নয়। "
            "বিশেষজ্ঞের পরামর্শ নিন।"
        )
    else:
        message = "রোগটি সফলভাবে শনাক্ত করা হয়েছে।"

    return {
        "disease": disease_bn,
        "confidence": round(confidence, 2),
        "needsExpertReview": needs_expert_review,
        "message": message,
    }


# =====================================================
# Rice Bangla Text Prediction API
# Existing endpoint kept for app compatibility
# =====================================================


@app.post("/predict/text")
def predict_rice_text(
    request: TextPredictionRequest,
    api_key: str = Security(verify_api_key),
):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="লক্ষণ লিখুন।",
        )

    return predict_with_text_model(
        text=text,
        model=rice_text_model,
        disease_mapping=RICE_DISEASE_BN,
    )


# =====================================================
# Eggplant Bangla Text Prediction API
# =====================================================


@app.post("/predict/eggplant/text")
def predict_eggplant_text(
    request: TextPredictionRequest,
    api_key: str = Security(verify_api_key),
):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="লক্ষণ লিখুন।",
        )

    return predict_with_text_model(
        text=text,
        model=eggplant_text_model,
        disease_mapping=EGGPLANT_DISEASE_BN,
    )
