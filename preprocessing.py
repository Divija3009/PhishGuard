import pandas as pd
import re
from urllib.parse import urlparse

# --------------------------
# Utility Feature Functions
# --------------------------

def get_domain(email):
    if pd.isna(email): return ''
    parts = email.split('@')
    return parts[1].strip() if len(parts) == 2 else ''

def contains_suspicious_keywords(text):
    keywords = ['account', 'verify', 'password', 'login', 'bank', 'click', 'urgent', 'invoice', 'refund']
    return any(word in str(text).lower() for word in keywords)

def is_display_name_mismatch(sender):
    match = re.match(r"(.*?)<(.+?)>", str(sender))
    if match:
        name, email = match.groups()
        return name.strip().lower() not in email.lower()
    return False

def unique_char_ratio(text):
    text = str(text)
    return len(set(text)) / len(text) if len(text) > 0 else 0

def extract_url_features_from_body(body):
    if not body:
        return 0
    urls = re.findall(r'https?://[^\s]+', body)
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account', 'bank']
    count = 0
    for url in urls:
        try:
            parsed = urlparse(url)
            if any(word in parsed.netloc for word in suspicious_keywords):
                count += 1
        except Exception:
            continue  # Skip malformed URLs
    return count

def safe_url_feature_extraction(body):
    try:
        return extract_url_features_from_body(body)
    except:
        return 0

# --------------------------
# Main Feature Engineering
# --------------------------

def engineer_features(df):
    df = df.copy()  # Prevent SettingWithCopyWarning

    # Drop rows with missing essential values
    df = df.dropna(subset=['sender', 'receiver', 'subject', 'body', 'label', 'urls', 'date'])

    # Convert 'date' and drop invalid
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date'])

    # Time-based features
    df.loc[:, 'hour'] = df['date'].dt.hour
    df.loc[:, 'sent_at_odd_hour'] = df['hour'].apply(lambda x: x < 6 or x > 22)

    # Domain and metadata features
    df.loc[:, 'sender_domain'] = df['sender'].apply(get_domain)
    df.loc[:, 'receiver_domain'] = df['receiver'].apply(get_domain)
    df.loc[:, 'domain_mismatch'] = df['sender_domain'] != df['receiver_domain']

    df.loc[:, 'subject_keywords'] = df['subject'].apply(contains_suspicious_keywords)
    df.loc[:, 'body_keywords'] = df['body'].apply(contains_suspicious_keywords)
    df.loc[:, 'display_name_mismatch'] = df['sender'].apply(is_display_name_mismatch)
    df.loc[:, 'subject_length'] = df['subject'].apply(lambda x: len(str(x)))
    df.loc[:, 'body_length'] = df['body'].apply(lambda x: len(str(x)))
    df.loc[:, 'unique_char_ratio'] = df['body'].apply(unique_char_ratio)

    return df
