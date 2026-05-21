# ==========================================================
# STREAMLIT APP
# File: app/streamlit_app.py
# ==========================================================


# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Diabetes Prediction System",

    page_icon="🩺",

    layout="centered"
)


# ==========================================================
# 3. LOAD MODEL
# ==========================================================

model = joblib.load(
    "models/best_tuned_model.pkl"
)


# ==========================================================
# 4. LOAD SCALER
# ==========================================================

scaler = joblib.load(
    "models/scaler.pkl"
)


# ==========================================================
# 5. LOAD MODEL NAME
# ==========================================================

with open(
    "models/best_tuned_model_name.txt",
    "r"
) as file:

    model_name = file.read()


# ==========================================================
# 6. APP TITLE
# ==========================================================

st.title("🩺 Diabetes Prediction System")

st.write(
    """
    This machine learning application predicts whether
    a patient is diabetic based on medical measurements.
    """
)

st.write(f"### Model Used: {model_name}")


# ==========================================================
# 7. SIDEBAR INPUTS
# ==========================================================

st.sidebar.header("Patient Information")


pregnancies = st.sidebar.number_input(

    "Pregnancies",

    min_value=0,

    max_value=20,

    value=1
)

glucose = st.sidebar.number_input(

    "Glucose",

    min_value=0,

    max_value=300,

    value=120
)

blood_pressure = st.sidebar.number_input(

    "Blood Pressure",

    min_value=0,

    max_value=150,

    value=70
)

skin_thickness = st.sidebar.number_input(

    "Skin Thickness",

    min_value=0,

    max_value=100,

    value=20
)

insulin = st.sidebar.number_input(

    "Insulin",

    min_value=0,

    max_value=900,

    value=79
)

bmi = st.sidebar.number_input(

    "BMI",

    min_value=0.0,

    max_value=70.0,

    value=30.0
)

dpf = st.sidebar.number_input(

    "Diabetes Pedigree Function",

    min_value=0.0,

    max_value=3.0,

    value=0.5
)

age = st.sidebar.number_input(

    "Age",

    min_value=1,

    max_value=120,

    value=30
)


# ==========================================================
# 8. CREATE INPUT DATAFRAME
# ==========================================================

input_data = pd.DataFrame({

    "Pregnancies": [pregnancies],

    "Glucose": [glucose],

    "BloodPressure": [blood_pressure],

    "SkinThickness": [skin_thickness],

    "Insulin": [insulin],

    "BMI": [bmi],

    "DiabetesPedigreeFunction": [dpf],

    "Age": [age]
})


# ==========================================================
# 9. DISPLAY INPUT DATA
# ==========================================================

st.subheader("Input Data")

st.dataframe(input_data)


# ==========================================================
# 10. MODELS REQUIRING SCALING
# ==========================================================

scaled_models = [

    "Logistic Regression",

    "KNN",

    "SVM"
]


# ==========================================================
# 11. APPLY SCALING
# ==========================================================

if model_name in scaled_models:

    scaled_input = scaler.transform(
        input_data
    )

    prediction_input = scaled_input

else:

    prediction_input = input_data


# ==========================================================
# 12. PREDICTION BUTTON
# ==========================================================

if st.button("Predict Diabetes"):

    # ------------------------------------------------------
    # MAKE PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(
        prediction_input
    )[0]

    probability = model.predict_proba(
        prediction_input
    )[:, 1][0]

    # ------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("Patient is likely Diabetic")

    else:

        st.success("Patient is likely Non-Diabetic")

    # ------------------------------------------------------
    # DISPLAY PROBABILITY
    # ------------------------------------------------------

    st.write(
        f"### Diabetes Probability: {probability:.4f}"
    )

    # ------------------------------------------------------
    # RISK LEVEL
    # ------------------------------------------------------

    if probability < 0.30:

        risk = "Low Risk"

    elif probability < 0.70:

        risk = "Moderate Risk"

    else:

        risk = "High Risk"

    st.write(f"### Risk Level: {risk}")


# ==========================================================
# 13. MODEL INFORMATION
# ==========================================================

st.subheader("Model Information")

st.write(f"Best Tuned Model: {model_name}")

st.write(
    """
    Features Used:
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age
    """
)


# ==========================================================
# 14. FOOTER
# ==========================================================

st.markdown("---")

st.write(
    "Machine Learning Diabetes Prediction Project"
)