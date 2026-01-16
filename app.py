# import streamlit as st
# import numpy as np

# st.set_page_config(
#     page_title="Bank Churn Predictor",
#     page_icon="🏦",
#     layout="wide"
# )

# # ---------- LOAD MODEL ----------
# from tensorflow.keras.models import load_model
# model = load_model("churn_model.keras", compile=False)


# # ---------- TITLE ----------
# st.markdown(
#     "<h1 style='text-align:center;'>🏦 Bank Churn Predictor</h1>",
#     unsafe_allow_html=True
# )

# st.markdown("---")

# # ---------- INPUT LAYOUT ----------
# col1, col2 = st.columns(2)

# with col1:
#     credit_score = st.number_input("Credit Score", 300, 900, 650)
#     geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     age = st.number_input("Age", 18, 100, 40)
#     tenure = st.number_input("Tenure (Years)", 0, 10, 3)

# with col2:
#     balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
#     num_products = st.selectbox("Num of Products", [1, 2, 3, 4])
#     has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
#     is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
#     salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# st.markdown("<br>", unsafe_allow_html=True)

# # ---------- MANUAL ENCODING (NO RETRAINING) ----------
# gender_male = 1 if gender == "Male" else 0
# geo_france = 1 if geography == "France" else 0
# geo_germany = 1 if geography == "Germany" else 0
# geo_spain = 1 if geography == "Spain" else 0
# credit_card = 1 if has_card == "Yes" else 0
# active = 1 if is_active == "Yes" else 0

# # ⚠️ MUST MATCH YOUR TRAINING ORDER (11 features)
# features = np.array([[
#     credit_score,
#     age,
#     tenure,
#     balance,
#     num_products,
#     credit_card,
#     active,
#     salary,
#     gender_male,
#     geo_france,
#     geo_germany
# ]])

# # ---------- PREDICTION ----------
# if st.button("Predict Churn"):
#     prob = float(np.ravel(model.predict(features))[0])
#     prediction = "Churn" if prob > 0.5 else "No Churn"

#     st.markdown("---")
#     col3, col4 = st.columns(2)

#     with col3:
#         st.subheader("Prediction")
#         st.markdown(
#             f"<h2 style='color:{'red' if prediction=='Churn' else 'green'};'>"
#             f"{prediction}</h2>",
#             unsafe_allow_html=True
#         )

#     with col4:
#         st.subheader("Probability")
#         st.markdown(
#             f"<h2>{prob*100:.2f}%</h2>",
#             unsafe_allow_html=True
#         )
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Prediction ( Deep Learning )", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction (Deep Learning) P_19")
st.write("Enter customer details and predict churn Prediction.")

# ✅ Load model
@st.cache_resource
def load_dl_model():
    return load_model("churn_model.keras")

model = load_dl_model()

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


# ✅ Manual Encoding (same as most churn datasets)
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# ✅ Create Input Array in correct order
# Typical order used in Churn Modelling dataset training:
# [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geo_Germany, Geo_Spain, Gender_Male]
X = np.array([[credit_score, age, tenure, balance, num_products,
               has_cr_card, is_active_member, estimated_salary,
               geo_germany, geo_spain, gender_male]])

st.write("✅ Input Array Preview:")
st.write(X)

if st.button("🔍 Predict Churn"):
    pred_prob = float(model.predict(X)[0][0])
    pred_class = 1 if pred_prob >= 0.5 else 0

    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Probability:** {pred_prob:.2f}")

    if pred_class == 1:
        st.error("⚠️ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is NOT likely to churn.")

