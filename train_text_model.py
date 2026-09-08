import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# Configuration
# =========================

DATASET_PATH = (
    "dataset/"
    "bangla_rice_disease_dataset_1800_with_split.csv"
)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bangla_text_model.joblib"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================
# Load Dataset
# =========================

df = pd.read_csv(DATASET_PATH)

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()


print(
    "Train samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)


# =========================
# Input / Label
# =========================

X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]


# =========================
# TF-IDF + Logistic Regression
# =========================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000,
            C=4.0
        )
    )
])


# =========================
# Train
# =========================

print("\nTraining started...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================
# Validation
# =========================

val_predictions = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_val,
        val_predictions,
        digits=4
    )
)


# =========================
# Save Model
# =========================

joblib.dump(
    model,
    MODEL_PATH
)


print(
    "\nModel saved at:",
    MODEL_PATH
)