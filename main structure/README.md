# Smart Feedback Classifier 🧠💬

A lightweight sentiment classification and tagging system for customer feedback from multiple sources (email, chat, reviews, etc). Trained using machine learning models and exposed via a REST API for real-time integration.

## 🔍 Problem Statement

Companies receive massive volumes of unstructured feedback from customers. This project builds a system that:
- Classifies feedback as positive/negative
- Tags intent (e.g., complaint, praise, pricing)
- Returns structured responses via API

## 🧰 Tech Stack

- Python, Pandas, Scikit-learn
- XGBoost / RandomForest
- FastAPI for model deployment
- MLflow / W&B for experiment tracking
- Docker (optional)
- Streamlit (optional for dashboard)

## 🚀 How to Run

```bash
# Create venv and install dependencies
pip install -r requirements.txt

# Train model
python model/train.py

# Run API
uvicorn api.main:app --reload
```

## 📊 Example Output

POST /predict

```json
{
  "text": "I’m very disappointed with the late delivery and poor packaging",
  "sentiment": "negative",
  "intent": "logistics_issue",
  "confidence": 0.94
}
```

## 📁 Project Structure

- `notebooks/`: EDA and data exploration
- `model/`: training and inference logic
- `api/`: FastAPI app to serve model
- `utils/`: helper functions

## 📊 Model Performance

After training, the sentiment analysis model achieved the following evaluation metrics on the test set:

* **Accuracy**: 0.7500
* **F1 Score**: 0.8148
* **ROC-AUC**: 0.8800