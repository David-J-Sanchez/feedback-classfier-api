"""
inference.py — Use trained model to make predictions
"""

import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("xgb_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def predict(text: str):
    # Convert input to vector
    X = vectorizer.transform([text])

    # Predict label and confidence
    label = model.predict(X)[0]
    prob = model.predict_proba(X).max()

    return {
        "input": text,
        "prediction": label,
        "confidence": round(float(prob), 3)
    }

# Example usage
if __name__ == "__main__":
    example = "This product arrived late and broken."
    result = predict(example)
    print(result)
