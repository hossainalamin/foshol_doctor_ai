import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path("models/bangla_text_model.joblib")
OUTPUT_PATH = Path("models/bangla_text_offline.json.gz")


def find_components(obj):
    vectorizer = None
    classifier = None

    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if isinstance(step, TfidfVectorizer):
                vectorizer = step
            elif isinstance(step, LogisticRegression):
                classifier = step

    elif isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, TfidfVectorizer):
                vectorizer = value
            elif isinstance(value, LogisticRegression):
                classifier = value

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, TfidfVectorizer):
                vectorizer = value
            elif isinstance(value, LogisticRegression):
                classifier = value

    # Support an object that directly exposes vectorizer/classifier-like attrs.
    if vectorizer is None:
        for name in ("vectorizer", "tfidf", "tfidf_vectorizer"):
            value = getattr(obj, name, None)
            if isinstance(value, TfidfVectorizer):
                vectorizer = value
                break

    if classifier is None:
        for name in ("classifier", "model", "clf", "logreg"):
            value = getattr(obj, name, None)
            if isinstance(value, LogisticRegression):
                classifier = value
                break

    if vectorizer is None or classifier is None:
        raise RuntimeError(
            "Could not find TfidfVectorizer + LogisticRegression inside the joblib model."
        )

    return vectorizer, classifier


def main():
    obj = joblib.load(MODEL_PATH)
    vectorizer, classifier = find_components(obj)

    if vectorizer.analyzer != "char_wb":
        raise RuntimeError(
            f"This Android implementation expects analyzer='char_wb', got {vectorizer.analyzer!r}"
        )

    if not hasattr(vectorizer, "idf_"):
        raise RuntimeError("The fitted TF-IDF vectorizer does not contain idf_.")

    solver = getattr(classifier, "solver", "")
    multi_class = getattr(classifier, "multi_class", "auto")
    probability_mode = (
        "ovr"
        if multi_class == "ovr" or (multi_class == "auto" and solver == "liblinear")
        else "softmax"
    )

    data = {
        "version": 1,
        "modelType": "tfidf_logistic_regression",
        "analyzer": "char_wb",
        "ngramMin": int(vectorizer.ngram_range[0]),
        "ngramMax": int(vectorizer.ngram_range[1]),
        "lowercase": bool(vectorizer.lowercase),
        "sublinearTf": bool(vectorizer.sublinear_tf),
        "norm": vectorizer.norm,
        "vocabulary": {str(k): int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf": np.asarray(vectorizer.idf_, dtype=np.float32).tolist(),
        "coef": np.asarray(classifier.coef_, dtype=np.float32).tolist(),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float32).tolist(),
        "classes": [str(x) for x in classifier.classes_.tolist()],
        "probabilityMode": probability_mode,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Created: {OUTPUT_PATH}")
    print(f"Classes: {data['classes']}")
    print(f"Features: {len(data['vocabulary'])}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
