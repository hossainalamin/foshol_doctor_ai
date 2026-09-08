import joblib

MODEL_PATH = "models/bangla_text_model.joblib"

model = joblib.load(MODEL_PATH)

print("Bangla text model loaded successfully.\n")

while True:

    text = input("বাংলায় লক্ষণ লিখুন (exit লিখলে বন্ধ হবে): ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    best_index = probabilities.argmax()
    confidence = probabilities[best_index] * 100

    print("\nPredicted Disease:", prediction)
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop predictions:")

    top_indices = probabilities.argsort()[::-1][:3]

    for index in top_indices:
        print(
            f"{classes[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )

    print("-" * 50)