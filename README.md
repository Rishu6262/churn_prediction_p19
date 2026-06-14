# 🧠 Customer Churn Prediction Using Deep Learning

---

🚀 Live Application

🔗 https://churnpredictionp19-evkjk4fzrarzujckjevh9h.streamlit.app/

Predict customer churn using Deep Learning and Artificial Neural Networks (ANN). The model analyzes customer demographics, banking behavior, and financial information to identify customers who are likely to leave the bank.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by businesses, especially in the banking and financial sector. Customer churn occurs when a customer stops using a company's products or services.

This project uses Deep Learning techniques to predict whether a customer is likely to leave the bank based on demographic, financial, and behavioral information.

The goal is to help organizations identify high-risk customers and take proactive actions to improve customer retention.

---

# ❓ Why I Chose This Project?

Customer retention is more cost-effective than acquiring new customers. Predicting customer churn is a real-world business problem that directly impacts company revenue and growth.

I chose this project to:

* Learn Deep Learning concepts.
* Work with real-world business data.
* Understand customer behavior patterns.
* Build an Artificial Neural Network (ANN).
* Apply predictive analytics in business intelligence.

---

# 🚀 Project Objectives

* Predict customer churn using Deep Learning.
* Analyze customer behavior patterns.
* Reduce customer attrition risk.
* Improve customer retention strategies.
* Build an end-to-end Deep Learning solution.

---

# 📊 Dataset Information

### Dataset Name

Customer Churn Modelling Dataset

### Total Records

* 10,000 Customers

### Total Features

* 14 Features

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 0     | Customer Stayed |
| 1     | Customer Left   |

---

## Features Description

| Feature         | Description              |
| --------------- | ------------------------ |
| CreditScore     | Customer Credit Score    |
| Geography       | Customer Country         |
| Gender          | Male/Female              |
| Age             | Customer Age             |
| Tenure          | Years with Bank          |
| Balance         | Bank Balance             |
| NumOfProducts   | Number of Bank Products  |
| HasCrCard       | Credit Card Holder       |
| IsActiveMember  | Active Membership Status |
| EstimatedSalary | Estimated Annual Salary  |
| Exited          | Customer Churn Status    |

---

# 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Keras
* Streamlit

---

# 📂 Project Structure

```bash
Customer_Churn_Prediction/
│
├── app.py
├── model.h5
├── scaler.pkl
├── Churn_Modelling.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── churn_prediction.ipynb
│
└── assets/
    └── screenshots/
```

---

# 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

### Customer Analysis

* Gender Distribution
* Geography Distribution
* Age Analysis

### Financial Analysis

* Credit Score Analysis
* Balance Distribution
* Estimated Salary Analysis

### Churn Analysis

* Churn vs Non-Churn Customers
* Active vs Inactive Members
* Product Usage Analysis

### Correlation Analysis

* Correlation Heatmap
* Feature Importance

---

# ⚙️ Data Preprocessing

### Data Cleaning

* Removed Unnecessary Columns
* Checked Missing Values
* Removed Duplicates

### Encoding

Categorical Features Converted Using:

* Label Encoding
* One-Hot Encoding

### Feature Scaling

Applied:

```python
StandardScaler()
```

to normalize numerical features.

---

# 🤖 Deep Learning Model

## Artificial Neural Network (ANN)

The model was built using TensorFlow and Keras.

### Architecture

Input Layer

↓

Dense Layer (ReLU)

↓

Dense Layer (ReLU)

↓

Dropout Layer

↓

Dense Layer (Sigmoid)

---

### Activation Functions

* ReLU
* Sigmoid

### Optimizer

```python
Adam
```

### Loss Function

```python
Binary Crossentropy
```

---

# 📈 Model Evaluation Metrics

The model was evaluated using:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures correctness of positive predictions.

### Recall

Measures ability to detect churn customers.

### F1 Score

Balances Precision and Recall.

### Confusion Matrix

Shows classification performance.

---

# 🏆 Model Performance

The Deep Learning model was trained and evaluated using train-test split.

Performance was measured based on:

* Accuracy
* Precision
* Recall
* F1 Score
* Validation Loss

The final trained model was saved for deployment.

---

# 💻 Streamlit Web Application

The project includes a Streamlit-based web application.

### User Inputs

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Membership
* Estimated Salary

### Output

* Customer Will Stay
* Customer Will Leave

with prediction probability.

---

# ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
joblib
```

---

# 🎯 Learning Outcomes

Through this project, I learned:

* Deep Learning Fundamentals
* Artificial Neural Networks (ANN)
* Data Preprocessing
* Feature Engineering
* Model Evaluation
* TensorFlow & Keras
* Streamlit Deployment
* Customer Analytics

---

# 🔮 Future Improvements

* Advanced Neural Networks
* Hyperparameter Tuning
* Real-Time Prediction API
* Explainable AI (XAI)
* Customer Retention Dashboard

---

# 📜 Disclaimer

This project is developed for educational and research purposes only.

The predictions generated by the model are based on historical customer data and should not be considered business decisions without further analysis.

---

# ✅ Conclusion

This project demonstrates how Deep Learning can be used to predict customer churn by analyzing customer demographics, banking behavior, and financial information. The ANN model helps identify customers who are likely to leave, enabling organizations to improve customer retention and make data-driven business decisions.

---

# 👨‍💻 Author

**Rishu Gurjar**

Aspiring Data Science | Machine Learning Enthusiast | Deep Learning Developer
