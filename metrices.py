import ast
import streamlit as st
import pandas as pd


st.set_page_config(page_title="📊 Model Performance Dashboard", layout="wide")

# -----------------------
# Load Metrics
# -----------------------
st.title("Email Phishing Detection")
st.title("📊 Model Evaluation")

try:
    df = pd.read_csv("model_metrics.csv")
except FileNotFoundError:
    st.error("❌ model_metrics.csv not found. Please run train_models.py first.")
    st.stop()

# Convert confusion matrix string back to list
df["Confusion Matrix"] = df["Confusion Matrix"].apply(lambda x: ast.literal_eval(x))

# -----------------------
# Metric Table
# -----------------------
st.subheader("📋 Overall Performance Table")
st.dataframe(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].set_index("Model"))