
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "stroke_prediction_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

features = joblib.load(
    "model_features.pkl"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title-box {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    text-align:center;
}

.result-safe {
    background:#dcfce7;
    padding:20px;
    border-radius:12px;
    color:#166534;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

.result-risk {
    background:#fee2e2;
    padding:20px;
    border-radius:12px;
    color:#991b1b;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("""
<div class="title-box">
    <h1>🩺 Stroke Prediction System</h1>
    <p>
        AI-Based Healthcare Prediction using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# INFO CARD
# =========================

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <h3>Dataset</h3>
        Healthcare Stroke Dataset
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <h3>Models</h3>
        Random Forest & Logistic Regression
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <h3>Target</h3>
        Stroke Risk Prediction
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# =========================
# INPUT SECTION
# =========================

st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        1,
        100,
        30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )

with col2:

    ever_married = st.selectbox(
        "Ever Married",
        ["No","Yes"]
    )

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children"
        ]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Urban","Rural"]
    )

    glucose = st.number_input(
        "Average Glucose Level",
        50.0,
        300.0,
        100.0
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    smoking = st.selectbox(
        "Smoking Status",
        [
            "never smoked",
            "formerly smoked",
            "smokes"
        ]
    )

# =========================
# ENCODING
# =========================

gender_map = {
    "Male":1,
    "Female":0
}

married_map = {
    "No":0,
    "Yes":1
}

work_map = {
    "Private":2,
    "Self-employed":3,
    "Govt_job":0,
    "children":1
}

residence_map = {
    "Urban":1,
    "Rural":0
}

smoking_map = {
    "never smoked":2,
    "formerly smoked":1,
    "smokes":3
}

# =========================
# PREDICT
# =========================

if st.button("🔍 Predict Stroke Risk"):

    data = pd.DataFrame([[
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        married_map[ever_married],
        work_map[work_type],
        residence_map[residence],
        glucose,
        bmi,
        smoking_map[smoking]
    ]], columns=features)

    # DEBUG
    data[numeric_cols] = scaler.transform(
    data[numeric_cols]
)

    numeric_cols = [
        'age',
        'avg_glucose_level',
        'bmi'
    ]

    data[numeric_cols] = scaler.transform(
        data[numeric_cols]
    )

    prediction = model.predict(data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0][1]
    else:
        probability = 0

    st.write("")
    st.subheader("📊 Prediction Result")

    st.progress(float(probability))

    st.metric(
        "Stroke Probability",
        f"{probability*100:.2f}%"
    )

    if prediction == 1:

        st.markdown(
        """
        <div class="result-risk">
        ⚠️ High Risk of Stroke
        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        """
        <div class="result-safe">
        ✅ Low Risk of Stroke
        </div>
        """,
        unsafe_allow_html=True
        )

st.write("")
st.write("---")
st.caption(
    "AI & Big Data Final Project 2026 | Stroke Prediction System"
)
