import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------- LOAD MODEL ----------
from tensorflow.keras.models import load_model
model = load_model("churn_model.keras", compile=False)


# ---------- TITLE ----------
st.markdown(
    "<h1 style='text-align:center;'>🏦 Bank Churn Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- INPUT LAYOUT ----------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 40)
    tenure = st.number_input("Tenure (Years)", 0, 10, 3)

with col2:
    balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
    num_products = st.selectbox("Num of Products", [1, 2, 3, 4])
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- MANUAL ENCODING (NO RETRAINING) ----------
gender_male = 1 if gender == "Male" else 0
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
credit_card = 1 if has_card == "Yes" else 0
active = 1 if is_active == "Yes" else 0

# ⚠️ MUST MATCH YOUR TRAINING ORDER (11 features)
features = np.array([[
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    credit_card,
    active,
    salary,
    gender_male,
    geo_france,
    geo_germany
]])

# ---------- PREDICTION ----------
if st.button("Predict Churn"):
    prob = float(np.ravel(model.predict(features))[0])
    prediction = "Churn" if prob > 0.5 else "No Churn"

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Prediction")
        st.markdown(
            f"<h2 style='color:{'red' if prediction=='Churn' else 'green'};'>"
            f"{prediction}</h2>",
            unsafe_allow_html=True
        )

    with col4:
        st.subheader("Probability")
        st.markdown(
            f"<h2>{prob*100:.2f}%</h2>",
            unsafe_allow_html=True
        )
