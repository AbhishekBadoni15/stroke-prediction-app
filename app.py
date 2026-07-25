import streamlit as st
import joblib
import pandas as pd

# Model Load
model = joblib.load("stroke_model.pkl")

st.set_page_config(page_title="Stroke Prediction", page_icon="❤️")

st.title("Stroke Prediction System")
st.write("Enter the patient details below.")
# User Inputs

age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension = st.selectbox("Hypertension", [0, 1])

heart_disease = st.selectbox("Heart Disease", [0, 1])

avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)

bmi = st.number_input("BMI", value=25.0)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])

ever_married = st.selectbox("Ever Married", ["No", "Yes"])

work_type = st.selectbox(
    "Work Type",
    ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]
)

residence_type = st.selectbox(
    "Residence Type",
    ["Rural", "Urban"]
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["Unknown", "formerly smoked", "never smoked", "smokes"]
)
# One-Hot Encoding

input_data = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,

    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Other": 1 if gender == "Other" else 0,

    "ever_married_Yes": 1 if ever_married == "Yes" else 0,

    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
    "work_type_children": 1 if work_type == "children" else 0,

    "Residence_type_Urban": 1 if residence_type == "Urban" else 0,

    "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
    "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
    "smoking_status_smokes": 1 if smoking_status == "smokes" else 0
}

input_df = pd.DataFrame([input_data])
if st.button("Predict Stroke"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

st.write(f"Stroke Probability: {probability[0][1]*100:.2f}%")

st.info("""
This prediction is generated using a machine learning model trained on historical data.
It is intended for educational purposes only and should not be considered a medical diagnosis.
""")