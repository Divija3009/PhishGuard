# 🧠 PhishGuard AI – Hybrid Machine Learning System for Email Phishing Detection  

## Overview  
**PhishGuard AI** is a powerful hybrid system that detects phishing emails using both **supervised** and **unsupervised machine learning techniques**.  
The system extracts features from email content and metadata (sender, subject, URLs, and timestamps) and classifies whether an email is **legitimate** or **phishing**.  

This project is built using **Python**, **scikit-learn**, and **Streamlit**, with custom preprocessing and feature-engineering pipelines for maximum flexibility and interpretability.

## 🎯 Key Objectives
- Detect phishing emails using **classification and anomaly detection** models.  
- Engineer features from email metadata (sender, domain mismatch, subject, URLs).  
- Train multiple ML models and evaluate their accuracy, precision, recall, and F1 score.  
- Provide a **Streamlit web app** for real-time email classification.

## Installation and Setup 
### 1. Clone the Repository   
```
git clone https://github.com/Divija3009/PhishGuard-AI.git
```
### 2. Create and Activate a Virtual Environment 
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

### 3.  Install Required Packages
pip install -r requirements.txt

### 4. Training the Models
Supervised Learning
```
python train_models.py
```
Unsupervised Learning
```
python train_unsupervised.py
```
### 5. Running the Streamlit App
```
streamlit run app.py
streamlit run app_unsupervised.py
``` 

