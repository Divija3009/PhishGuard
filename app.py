import streamlit as st
import pandas as pd
import joblib
import re
from email import policy
from email.parser import BytesParser
from preprocessing import (
    get_domain, contains_suspicious_keywords, is_display_name_mismatch,
    unique_char_ratio, extract_url_features_from_body
)

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Phishing Detection App", page_icon="📨")
st.title("📨 Email Phishing Detection")

# -------------------------------
# Load models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("models/classifier_logistic.pkl"),
    "Random Forest": joblib.load("models/classifier_rf.pkl"),
    "Ridge Classifier": joblib.load("models/classifier_ridge.pkl")
}

# -------------------------------
# Helper: Parse .eml file
# -------------------------------
def parse_eml(file):
    msg = BytesParser(policy=policy.default).parse(file)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                except:
                    body = ""
                break
    else:
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except:
            body = ""
    return {
        "sender": msg.get("From", ""),
        "receiver": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "date": msg.get("Date", ""),
        "body": body
    }

# -------------------------------
# Upload & Process Email
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an .eml file", type=["eml"])

if uploaded_file:
    email = parse_eml(uploaded_file)

    # Time features
    try:
        date = pd.to_datetime(email['date'], errors='coerce', utc=True)
        hour = date.hour if pd.notna(date) else 12
        sent_at_odd_hour = hour < 6 or hour > 22
    except:
        hour = 12
        sent_at_odd_hour = False

    # Features
    features = {
        'combined_text': f"{email['sender']} {email['receiver']} {email['subject']}",
        'url_suspicious_count': extract_url_features_from_body(email['body']),
        'domain_mismatch': get_domain(email['sender']) != get_domain(email['receiver']),
        'subject_keywords': contains_suspicious_keywords(email['subject']),
        'body_keywords': contains_suspicious_keywords(email['body']),
        'display_name_mismatch': is_display_name_mismatch(email['sender']),
        'subject_length': len(str(email['subject'])),
        'body_length': len(str(email['body'])),
        'sent_at_odd_hour': sent_at_odd_hour,
        'unique_char_ratio': unique_char_ratio(email['body'])
    }

    input_df = pd.DataFrame([features])

    # -------------------------------
    # Show Results
    # -------------------------------

    st.subheader("🔍 Model Predictions")
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        label = "🚨 Phishing" if pred == 1 else "✅ Legitimate"

        st.write(f"**{name}** — {label}")
        
    
    st.subheader("📨 Extracted Email Metadata")
    st.json(email)
