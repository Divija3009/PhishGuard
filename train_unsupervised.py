import os
import pandas as pd
import re
from sklearn.svm import OneClassSVM
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def extract_features(msg):
    features = {
        'subject_empty': 0,
        'message_id_valid': 1,
        'from_xfrom_mismatch': 0,
        'to_xto_mismatch': 0,
        'x_cc_or_bcc_used': 0,
        'x_origin_inconsistent': 0,
    }

    data = {
        'Message-ID': '',
        'From': '',
        'To': '',
        'Subject': '',
        'X-From': '',
        'X-To': '',
        'X-cc': '',
        'X-bcc': '',
        'X-Origin': ''
    }

    for line in msg.split("\n"):
        for key in data.keys():
            if line.lower().startswith(key.lower() + ":"):
                data[key] = line.split(":", 1)[1].strip()

    # Heuristic 1
    if data['Subject'] == '':
        features['subject_empty'] = 1

    # Heuristic 2
    if not re.match(r"^<.*@.*>$", data['Message-ID']):
        features['message_id_valid'] = 0

    # Heuristic 3
    if data['X-From'] and data['From'] and data['X-From'].lower() not in data['From'].lower():
        features['from_xfrom_mismatch'] = 1

    # Heuristic 4
    if data['X-To'] and data['To'] and data['X-To'].lower() not in data['To'].lower():
        features['to_xto_mismatch'] = 1

    # Heuristic 5
    if data['X-cc'] or data['X-bcc']:
        features['x_cc_or_bcc_used'] = 1

    # Heuristic 6
    from_domain = re.search(r"@([\w\.-]+)", data['From'])
    if from_domain and data['X-Origin']:
        if from_domain.group(1).lower() not in data['X-Origin'].lower():
            features['x_origin_inconsistent'] = 1

    return pd.Series(features)


df = pd.read_csv('data/emails.csv') 
n_samples = 100000

df = df.sample(n=n_samples,replace = True, random_state=63)
df = df.reset_index(drop=True)

heuristics_df = df['message'].apply(extract_features)

# === Train Models ===
models = {
    "ocsvm": OneClassSVM(kernel='rbf', nu=0.001, gamma='scale'),
    "lof": LocalOutlierFactor(n_neighbors=20, novelty=True),
    "isolation_forest": IsolationForest(contamination=0.005, random_state=42)
}

# === Save Directory ===
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# === Fit and Save Models ===
for name, model in models.items():
    model.fit(heuristics_df)
    joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

print(f"✅ All models saved to: {model_dir}/")