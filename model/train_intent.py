from utils.helpers import clean
import pandas as pd, os, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

# 1) Load CSV -------------------------------------------------
file_path = os.path.join(os.path.dirname(__file__), "..", "data", "feedback_with_intent.csv")
df = pd.read_csv(file_path, delimiter=';') # Explicitly setting semicolon delimiter

# 2) Basic cleaning ------------------------------------------
df["Text"] = df["Text"].astype(str).apply(clean)
df["Intent"] = df["Intent"].str.lower().str.strip()
mapping = {"praise": 1, "complaint": 0}
df["Intent"] = df["Intent"].map(mapping)
df.dropna(subset=["Text", "Intent"], inplace=True)

print(df["Intent"].value_counts())     # check balance

X, y = df["Text"], df["Intent"]

# 3) TF‑IDF ---------------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000,
                             ngram_range=(1,2),
                             stop_words="english",
                             min_df=2)
X_vec = vectorizer.fit_transform(X)
print("TRAIN ▶ TF-IDF Intent vocab size:", len(vectorizer.vocabulary_))

# 4) Simple baseline model -----------------------------------
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_cv = cross_val_score(log_reg, X_vec, y, cv=cv, scoring="f1").mean()
print(f"Baseline LogisticRegression F1 (5‑fold): {f1_cv:.3f}")

# 5) Train XGBoost with early stopping -----------------------
X_train, X_test, y_train, y_test = \
    train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

xgb = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=y.value_counts()[0]/y.value_counts()[1]  # balance
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:,1]

print("Acc:", accuracy_score(y_test, y_pred))
print("F1 :", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Confusion\n", confusion_matrix(y_test, y_pred))

# 6) Save artifacts ------------------------------------------
art_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(art_dir, exist_ok=True)
joblib.dump(xgb,  f"{art_dir}/intent_model.joblib")
joblib.dump(vectorizer, f"{art_dir}/intent_vectorizer.joblib")
print("✅  Saved model & vectorizer")
