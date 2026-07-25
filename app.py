import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stroke Prediction", page_icon="❤️")

# Model Load
model = joblib.load("stroke_model.pkl")

# Dataset Load
df = pd.read_csv("stroke_cleaned.csv")

# Sidebar
st.sidebar.title("🏠 Menu")

option = st.sidebar.radio(
    "Select Page",
    ["Prediction", "Dataset", "Visualizations", "About"]
)
if option == "Prediction":

    st.title("❤️ Stroke Prediction System")
    st.write("Enter the patient details below.")

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


elif option == "Dataset":

    st.subheader("Stroke Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Summary")
    st.write(df.describe())

elif option == "Visualizations":
    st.subheader("Stroke Distribution")

    fig, ax = plt.subplots()
    df["stroke"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["No Stroke", "Stroke"],
    startangle=90,
    ax=ax
)
    ax.set_ylabel("")
    st.pyplot(fig)
    
    # Age Distribution (Histogram)
    fig, ax = plt.subplots()
    df["age"].hist(bins=20, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # BMI Distribution
    st.subheader("BMI Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(df["bmi"].dropna())
    ax.set_title("BMI Distribution")
    ax.set_ylabel("BMI")
    ax.set_xticks([])
    st.pyplot(fig)

# Glucose Distribution
    fig, ax = plt.subplots()
    df["avg_glucose_level"].hist(bins=20, ax=ax)
    ax.set_title("Average Glucose Distribution")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif option == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
 ❤️ Stroke Prediction System

This application uses a Machine Learning model to predict the likelihood of stroke based on a patient's health information.

 🚀 Features
- Predict Stroke Risk
- View Stroke Dataset
- Interactive Data Visualizations
- Correlation Heatmap
- User-Friendly Interface

 🛠️ Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Joblib

 📊 Dataset
The application is trained on a cleaned stroke dataset containing patient health information such as age, BMI, glucose level, hypertension, heart disease, smoking status, work type, residence type, and gender.

 ⚠️ Disclaimer
This application is created for educational purposes only. The prediction should not be considered as professional medical advice.

---
**Developed By:** Abhishek

**Project:** Stroke Prediction Using Machine Learning
""")
