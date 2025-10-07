import streamlit as st
import pandas as pd
import joblib
import re
import email

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Phishing Detection App", page_icon="📨")
st.title("📨 Email Phishing Detection")

# -------------------------------
# Helper: Extract email from dataset
# -------------------------------

def get_email(eml_file):

    raw_email = eml_file.read().decode("utf-8")  # read bytes and decode to string
    msg = email.message_from_string(raw_email)
    return msg

    # # Read the .eml file using the email package
    # with open(eml_file_path, 'r') as file:
    #     raw_email = file.read()

    # # Parse the email using email.message_from_string
    # msg = email.message_from_string(raw_email)
    # return msg

# -------------------------------
# Helper: Extract features from email
# -------------------------------

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

    # Iterate over lines and assign values to the corresponding data
    for line in msg.split("\n"):
        for key in data.keys():
            if line.lower().startswith(key.lower() + ":"):
                data[key] = line.split(":", 1)[1].strip()

    # Heuristic 1: Check if the subject is empty
    if data['Subject'] == '':
        features['subject_empty'] = 1

    # Heuristic 2: Check if the Message-ID is valid
    if not re.match(r"^<.*@.*>$", data['Message-ID']):
        features['message_id_valid'] = 0

    # Heuristic 3: Check if "From" and "X-From" are inconsistent
    if data['X-From'] and data['From'] and data['X-From'].lower() not in data['From'].lower():
        features['from_xfrom_mismatch'] = 1

    # Heuristic 4: Check if "To" and "X-To" are inconsistent
    if data['X-To'] and data['To'] and data['X-To'].lower() not in data['To'].lower():
        features['to_xto_mismatch'] = 1

    # Heuristic 5: Check if "X-cc" or "X-bcc" are used
    if data['X-cc'] or data['X-bcc']:
        features['x_cc_or_bcc_used'] = 1

    # Heuristic 6: Check if "From" domain and "X-Origin" are inconsistent
    from_domain = re.search(r"@([\w\.-]+)", data['From'])
    if from_domain and data['X-Origin']:
        if from_domain.group(1).lower() not in data['X-Origin'].lower():
            features['x_origin_inconsistent'] = 1

    return pd.Series(features)

# -------------------------------
# Helper: Classify .eml file
# -------------------------------

# Function to classify a given .eml file
def classify_email(msg, ocsvm_model):

    # Extract headers from the email message
    email_headers = {
        'Message-ID': msg.get('Message-ID', ''),
        'From': msg.get('From', ''),
        'To': msg.get('To', ''),
        'Subject': msg.get('Subject', ''),
        'X-From': msg.get('X-From', ''),
        'X-To': msg.get('X-To', ''),
        'X-cc': msg.get('X-cc', ''),
        'X-bcc': msg.get('X-bcc', ''),
        'X-Origin': msg.get('X-Origin', '')
    }

    # Call the extract_features function with the parsed headers as input
    features_df = extract_features("\n".join(f"{k}: {v}" for k, v in email_headers.items())).to_frame().T

    # Check if the model is properly trained
    # if hasattr(ocsvm_model, 'support_vectors_'):
    #     print("Model is trained, proceeding with prediction.")
    # else:
    #     print("Error: Model is not properly trained. Please train the model first.")

    # Use the trained One-Class SVM model to predict if the email is phishing
    prediction = ocsvm_model.predict(features_df)

    # Return phishing flag (1 for phishing, 0 for not phishing)
    return 1 if prediction == -1 else 0

# -------------------------------
# Upload & Process Email
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an .eml file", type=["eml"])

if uploaded_file:
    email = get_email(uploaded_file)

    # Load models
    ocsvm_model = joblib.load("models/ocsvm.pkl")
    lof_model = joblib.load("models/lof.pkl")
    iso_model = joblib.load("models/isolation_forest.pkl")

    # Predict using all models
    result_ocsvm = classify_email(email, ocsvm_model)
    result_lof = classify_email(email, lof_model)
    result_iso = classify_email(email, iso_model)

    # -------------------------------
    # Show Results
    # -------------------------------
    st.subheader("🔍 Model Predictions")

    def label_result(result):
        return "🚨 Phishing" if result == 1 else "✅ Legitimate"

    st.write(f"**One-Class SVM** — {label_result(result_ocsvm)}")
    st.write(f"**Local Outlier Factor** — {label_result(result_lof)}")
    st.write(f"**Isolation Forest** — {label_result(result_iso)}")

    st.subheader("📨 Extracted Email Metadata")
    st.text(email.as_string())
    