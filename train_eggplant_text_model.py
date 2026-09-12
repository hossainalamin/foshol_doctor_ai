import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "dataset/eggplant_clean/eggplant_bangla_text_dataset_2100.csv"
MODEL_PATH = "models/bangla_eggplant_text_model.joblib"


# =========================
# Load Dataset
# =========================

df = pd.read_csv(DATA_PATH)

print("Total samples:", len(df))
print(df["label"].value_counts())


# =========================
# Split
# =========================

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

X_train = train_df["text_bn"]
y_train = train_df["label"]

X_val = val_df["text_bn"]
y_val = val_df["label"]

X_test = test_df["text_bn"]
y_test = test_df["label"]


# =========================
# Model
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
            C=4
        )
    )
])


# =========================
# Train
# =========================

print("\nTraining...")

model.fit(
    X_train,
    y_train
)


# =========================
# Validation
# =========================

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_val,
        val_pred
    )
)


# =========================
# Test
# =========================

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test,
        test_pred
    )
)


# =========================
# Save
# =========================

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved: {MODEL_PATH}"
)