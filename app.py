import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Churn Prediction (DL)", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction (Deep Learning)")
st.write("Enter customer details and predict whether the customer will churn or not.")

# =========================
# Load Model & Preprocessor
# =========================
@st.cache_resource
def load_all():
    model = load_model("churn_model.keras")

    with open("Churn_pred.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor

model, preprocessor = load_all()

# =========================
# Input UI
# =========================
st.subheader("🧾 Customer Information")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (Years)", 0, 10, 5)

balance = st.number_input("Balance", min_value=0.0,
