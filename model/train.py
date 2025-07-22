"""
train.py - Model Training Script

This script loads the dataset, preprocesses it, trains a classification model (e.g., XGBoost),
evaluates performance using metrics (Accuracy, F1, ROC-AUC), and saves the trained model.
"""

import os
import sys

# Get the directory of the current script (train.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root directory
project_root_dir = os.path.join(current_script_dir, os.pardir)
# Add the project root to sys.path
sys.path.insert(0, project_root_dir)

# Now you can add your other imports
# print("DEBUG: sys.path =", sys.path) # You can keep this for debugging, or remove after it works
import sys
print("DEBUG: sys.path =", sys.path)
from utils.helpers import clean
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

# 1. Load dataset (replace this with your actual CSV path)
df = pd.read_csv("data/feedback.csv", delimiter=';')

# 2. Preprocess - simple example (adapt this to your data)
# First, ensure the 'Sentiment' column is treated as string, strip whitespace, and convert to lowercase
# This will turn " Positive " into "positive" and " Negative " into "negative"
df["Text"] = df["Text"].astype(str).apply(clean)
df["Sentiment"] = df["Sentiment"].astype(str).apply(clean)

# Then, drop rows where 'Text' or the now cleaned 'Sentiment' are missing
df.dropna(subset=['Text', 'Sentiment'], inplace=True)

# Now map the cleaned sentiment values
X = df['Text']  # Feature
y = df['Sentiment'].map({'positive': 1, 'negative': 0})  # Binary target
print("DEBUG: Unique values in 'y' after mapping:", y.unique())
print("DEBUG: Number of NaN values in 'y' after mapping:", y.isnull().sum())

# 3. Feature Extraction (simple TF-IDF for text)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)
print("TRAIN ▶ TF-IDF Sentiment vocab size:", len(vectorizer.vocabulary_))
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# 5. Model Training
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_proba = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")

# 7. Save model and vectorizer
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(vectorizer, "artifacts/vectorizer.joblib")
