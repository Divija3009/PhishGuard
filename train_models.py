import pandas as pd
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from preprocessing import engineer_features, extract_url_features_from_body

# ----------------------------------------
# Load and preprocess data
# ----------------------------------------
df = pd.read_csv("data/CEAS_08.csv")
df = engineer_features(df)

df['url_suspicious_count'] = df['body'].apply(lambda x: extract_url_features_from_body(x) if pd.notna(x) else 0)
df['combined_text'] = df['sender'].fillna('') + ' ' + df['receiver'].fillna('') + ' ' + df['subject'].fillna('')

# Define features
meta_features = [
    'url_suspicious_count', 'domain_mismatch', 'subject_keywords', 'body_keywords',
    'display_name_mismatch', 'subject_length', 'body_length',
    'sent_at_odd_hour', 'unique_char_ratio'
]

X = df[['combined_text'] + meta_features]
y = df['label']

# ----------------------------------------
# Train/test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------
# Preprocessor: TF-IDF + Scaler
# ----------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("text", TfidfVectorizer(), "combined_text"),
    ("meta", StandardScaler(), meta_features)
])

# ----------------------------------------
# Train 3 models
# ----------------------------------------
models = {
    "classifier_logistic.pkl": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "classifier_rf.pkl": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100))
    ]),
    "classifier_ridge.pkl": Pipeline([
        ("pre", preprocessor),
        ("clf", RidgeClassifier())
    ])
}

# ----------------------------------------
# Train, evaluate, save
# ----------------------------------------
os.makedirs("models", exist_ok=True)

for filename, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n✅ {filename.replace('classifier_', '').replace('.pkl', '').title()} Results:")
    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, f"models/{filename}")

metrics_list = []

for filename, pipeline in models.items():
    model_name = filename.replace("classifier_", "").replace(".pkl", "").title()

    # Train & predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Print detailed classification report
    print(f"\n✅ {model_name} Results:")
    print(classification_report(y_test, y_pred))

    # Collect metrics
    metrics_list.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()  # convert to JSON-serializable
    })

    # Save model
    joblib.dump(pipeline, f"models/{filename}")

# Save metrics to CSV
pd.DataFrame(metrics_list).to_csv("model_metrics.csv", index=False)
print("\n📁 Metrics saved to model_metrics.csv")