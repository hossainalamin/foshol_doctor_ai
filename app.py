from io import BytesIO
import os
import secrets

import joblib
import torch
import torch.nn as nn
import cv2
import numpy as np

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Security
)

from fastapi.security import APIKeyHeader

from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()


# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="Fasol Doctor AI API",
    version="1.0",
    description="AI API for rice disease prediction using image and Bangla text."
)


# =====================================================
# Disease Bangla Mapping
# =====================================================

DISEASE_BN = {
    "Blast": "ব্লাস্ট",
    "Brown spot": "ব্রাউন স্পট",
    "Healthy": "সুস্থ",
    "Leaf smut": "লিফ স্মাট",
    "Rice Tungro": "রাইস টুংরো",
    "Sheath blight": "শীথ ব্লাইট"
}


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

IMAGE_MODEL_PATH = (
    "models/mobilenetv3_finetuned_best.pth"
)

TEXT_MODEL_PATH = (
    "models/bangla_text_model.joblib"
)

CONFIDENCE_THRESHOLD = 70.0


# =====================================================
# API Key Configuration
# =====================================================

API_KEY = os.getenv(
    "FASOL_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "FASOL_API_KEY is not configured. "
        "Please add it to the .env file."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str = Security(
        api_key_header
    )
):

    if (
        api_key is None
        or not secrets.compare_digest(
            api_key,
            API_KEY
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="অবৈধ অথবা অনুপস্থিত API Key।"
        )

    return api_key
# =====================================================
# Check Image Quality
# =====================================================
def check_image_quality(pil_image):

    image = np.array(pil_image)

    # Minimum size
    height, width = image.shape[:2]

    if width < 224 or height < 224:
        return False, "ছবির resolution খুব কম। পরিষ্কার ছবি তুলুন।"

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Blur check
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < 80:
        return False, "ছবিটি অস্পষ্ট। আবার পরিষ্কার ছবি তুলুন।"

    # Brightness check
    brightness = gray.mean()

    if brightness < 45:
        return False, "ছবিটি অনেক অন্ধকার। আলোতে আবার ছবি তুলুন।"

    if brightness > 225:
        return False, "ছবিটি অতিরিক্ত উজ্জ্বল। আবার ছবি তুলুন।"

    return True, None
# =====================================================
# Load Image Model
# =====================================================

checkpoint = torch.load(
    IMAGE_MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint[
    "class_names"
]

IMAGE_SIZE = checkpoint.get(
    "image_size",
    224
)


image_model = (
    models.mobilenet_v3_small(
        weights=None
    )
)


image_model.classifier[3] = (
    nn.Linear(
        image_model
        .classifier[3]
        .in_features,

        len(class_names)
    )
)


image_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


image_model = image_model.to(
    DEVICE
)

image_model.eval()


print(
    "Image AI model loaded successfully"
)

print(
    "Device:",
    DEVICE
)

print(
    "Classes:",
    class_names
)


# =====================================================
# Load Bangla Text Model
# =====================================================

text_model = joblib.load(
    TEXT_MODEL_PATH
)


print(
    "Bangla text model loaded successfully"
)


# =====================================================
# Image Preprocessing
# =====================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
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
# Image Prediction API
# =====================================================

@app.post("/predict/image")
async def predict_image(

    image: UploadFile = File(...),

    api_key: str = Security(
        verify_api_key
    )

):

    # -----------------------------------------
    # Validate Image
    # -----------------------------------------

    if (
        image.content_type is None
        or not image.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "অনুগ্রহ করে একটি সঠিক "
                "ছবি আপলোড করুন।"
            )
        )


    # -----------------------------------------
    # Read Image
    # -----------------------------------------

    try:
        contents = await image.read()

        pil_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="ছবিটি পড়া সম্ভব হয়নি।"
        ) from exc

    # Quality check MUST be outside try/except
    is_valid, error_message = check_image_quality(
        pil_image
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_message
        )


    # -----------------------------------------
    # Image Preprocessing
    # -----------------------------------------

    input_tensor = (
        image_transform(
            pil_image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    with torch.no_grad():

        output = image_model(
            input_tensor
        )


        probabilities = (
            torch.softmax(
                output,
                dim=1
            )[0]
        )


    # -----------------------------------------
    # Best Prediction
    # -----------------------------------------

    best_index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    disease_en = str(
        class_names[
            best_index
        ]
    )


    disease_bn = DISEASE_BN.get(
        disease_en,
        disease_en
    )


    confidence = float(
        probabilities[
            best_index
        ].item()
        * 100
    )


    needs_expert_review = bool(
        confidence
        < CONFIDENCE_THRESHOLD
    )


    # -----------------------------------------
    # Message
    # -----------------------------------------

    if needs_expert_review:

        message = (
            "রোগ শনাক্তকরণে AI যথেষ্ট নিশ্চিত নয়। "
            "বিশেষজ্ঞের পরামর্শ নিন।"
        )

    else:

        message = (
            "রোগটি সফলভাবে শনাক্ত করা হয়েছে।"
        )


    # -----------------------------------------
    # Response
    # -----------------------------------------

    return {

        "disease": disease_bn,

        "confidence": round(
            confidence,
            2
        ),

        "needsExpertReview":
            needs_expert_review,

        "message": message
    }


# =====================================================
# Bangla Text Prediction API
# =====================================================

@app.post("/predict/text")
def predict_text(

    request: TextPredictionRequest,

    api_key: str = Security(
        verify_api_key
    )

):

    # -----------------------------------------
    # Validate Text
    # -----------------------------------------

    text = request.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,
            detail="লক্ষণ লিখুন।"
        )


    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    probabilities = (
        text_model.predict_proba(
            [text]
        )[0]
    )


    classes = (
        text_model.classes_
    )


    best_index = int(
        probabilities.argmax()
    )


    disease_en = str(
        classes[
            best_index
        ]
    )


    disease_bn = DISEASE_BN.get(
        disease_en,
        disease_en
    )


    confidence = float(
        probabilities[
            best_index
        ]
        * 100
    )


    needs_expert_review = bool(
        confidence
        < CONFIDENCE_THRESHOLD
    )


    # -----------------------------------------
    # Message
    # -----------------------------------------

    if needs_expert_review:

        message = (
            "রোগ শনাক্তকরণে AI যথেষ্ট নিশ্চিত নয়। "
            "বিশেষজ্ঞের পরামর্শ নিন।"
        )

    else:

        message = (
            "রোগটি সফলভাবে শনাক্ত করা হয়েছে।"
        )


    # -----------------------------------------
    # Response
    # -----------------------------------------

    return {

        "disease": disease_bn,

        "confidence": round(
            confidence,
            2
        ),

        "needsExpertReview":
            needs_expert_review,

        "message": message
    }