import os
import sys # Added for sys.path modification
import hashlib
import joblib
from fastapi import FastAPI
from pydantic import BaseModel # Ensure this is imported

# --- CRITICAL FIX: Add project root to sys.path for module discovery ---
# Get the directory of the current script (main.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root directory (V1- (Sent-Analys))
project_root_dir = os.path.join(current_script_dir, os.pardir)
# Add the project root to sys.path
sys.path.insert(0, project_root_dir)

# Now, import local modules
from utils.helpers import clean # Moved after sys.path modification

def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# --- Model Loading (Updated to load both models) ---
# Construct paths to the artifacts directory relative to the script's location
base_dir = os.path.dirname(__file__)
artifacts_dir = os.path.join(base_dir, "..", "artifacts")

# Load Sentiment Model and Vectorizer
sentiment_model_path = os.path.join(artifacts_dir, "sentiment_model.joblib")
sentiment_vectorizer_path = os.path.join(artifacts_dir, "sentiment_vectorizer.joblib")

# Load Intent Model and Vectorizer
intent_model_path = os.path.join(artifacts_dir, "intent_model.joblib")
intent_vectorizer_path = os.path.join(artifacts_dir, "intent_vectorizer.joblib")

# Check if files exist before loading (good practice)
try:
    sentiment_model = joblib.load(sentiment_model_path)
    sentiment_vectorizer = joblib.load(sentiment_vectorizer_path)
    intent_model = joblib.load(intent_model_path)
    intent_vectorizer = joblib.load(intent_vectorizer_path)
    print("All models and vectorizers loaded successfully!")
    # >>> CORRECTED INDENTATION FOR DEBUG PRINTS <<<
    print("DEBUG ▶ sentiment vec vocab size:", len(sentiment_vectorizer.vocabulary_))
    print("DEBUG ▶ intent    vec vocab size:", len(intent_vectorizer.vocabulary_))
    print("DEBUG ▶ sentiment model hash:", md5(sentiment_model_path))
    print("DEBUG ▶ intent    model hash:", md5(intent_model_path))
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Please ensure train.py and train_intent.py have been run and generated the .joblib files in the 'artifacts' directory.")
    # Exit or raise an exception to prevent API from starting without models
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Customer Feedback Analysis API",
    description="API for classifying sentiment and intent from customer feedback.",
    version="0.1.0"
)

# --- Define Pydantic Models for Request and Response ---
# This defines the structure of the incoming request body
class TextInput(BaseModel):
    text: str

# This defines the structure of the outgoing response
class PredictionOutput(BaseModel):
    input_text: str
    sentiment: str
    sentiment_confidence: float # Renamed for clarity if you have multiple confidences
    intent: str
    intent_confidence: float

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Customer Feedback Analysis API! Visit /docs for more information."}

@app.post("/predict_sentiment_and_intent", response_model=PredictionOutput)
async def predict_sentiment_and_intent(item: TextInput):
    # >>> CORRECTED INDENTATION FOR ENTIRE FUNCTION BODY <<<
    # --- Preprocess the input text consistently ---
    preprocessed_text = clean(item.text) # Using the new clean() helper

    # --- DEBUG PRINTS for preprocessing ---
    print(f"\nDEBUG: Original Input Text: '{item.text}'")
    print(f"DEBUG: Preprocessed Text: '{preprocessed_text}'")

    # --- Sentiment Prediction ---
    text_processed_sentiment = sentiment_vectorizer.transform([preprocessed_text])
    sentiment_prediction_proba = sentiment_model.predict_proba(text_processed_sentiment)[0]
    s_raw = sentiment_model.predict(text_processed_sentiment)[0] # Get raw prediction

    # --- DEBUG PRINTS for Sentiment ---
    print(f"DEBUG ▶ Sent raw class = {s_raw}, probs = {sentiment_prediction_proba}")
    print(f"DEBUG ▶ Sentiment Model Classes (model.classes_): {sentiment_model.classes_}") # Helpful to see internal order

    # Refactor sentiment label map
    sent_label_map = {0: "negative", 1: "positive"} # Explicit mapping
    predicted_sentiment = sent_label_map[s_raw]
    sentiment_confidence = sentiment_prediction_proba.max() # Still use max prob for confidence

    # --- Intent Prediction ---
    text_processed_intent = intent_vectorizer.transform([preprocessed_text])
    intent_prediction_proba = intent_model.predict_proba(text_processed_intent)[0]
    i_raw = intent_model.predict(text_processed_intent)[0] # Get raw prediction

    # --- DEBUG PRINTS for Intent ---
    # Corrected typo: added { } around intent_prediction_proba
    print(f"DEBUG ▶ Intent raw class = {i_raw}, probs = {intent_prediction_proba}")
    print(f"DEBUG ▶ Intent Model Classes (model.classes_): {intent_model.classes_}") # Helpful to see internal order

    # Refactor intent label map
    intent_label_map = {0: "complaint", 1: "praise"} # Explicit mapping
    predicted_intent = intent_label_map[i_raw]
    intent_confidence = intent_prediction_proba.max() # Still use max prob for confidence

    return PredictionOutput(
        input_text=item.text,
        sentiment=predicted_sentiment,
        sentiment_confidence=sentiment_confidence,
        intent=predicted_intent,
        intent_confidence=intent_confidence
    )