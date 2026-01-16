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

balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)

has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=80000.0)

# =========================
# Prepare DataFrame
# =========================
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

st.write("✅ Input Preview:")
st.dataframe(input_data)

# =========================
# Predict Button
# =========================
if st.button("🔍 Predict Churn"):
    try:
        # transform input
        X = preprocessor.transform(input_data)

        # Predict (DL output probability)
        pred_prob = model.predict(X)[0][0]
        pred_class = 1 if pred_prob >= 0.5 else 0

        st.subheader("📌 Prediction Result")
        st.write(f"**Churn Probability:** {pred_prob:.2f}")

        if pred_class == 1:
            st.error("⚠️ Customer is likely to CHURN!")
        else:
            st.success("✅ Customer is NOT likely to churn.")

    except Exception as e:
        st.warning("Your preprocessor file may not support `.transform()` directly.")
        st.error(f"Error: {e}")
        st.info(
            "If you want, I can fix this based on what exactly is inside `Churn_pred.pkl` "
            "(scaler only? encoder? pipeline?)."
        )
