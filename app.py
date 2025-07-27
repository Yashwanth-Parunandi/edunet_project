import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("💼 Employee Salary Prediction")

# User Inputs
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Senior Manager",
                                 "Sales Associate", "Director"])  # Update if you have more
experience = st.slider("Years of Experience", 0, 40, 5)

# Create input DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job,
    "Years of Experience": experience
}])

# Encode input and predict
encoded_input = encoder.transform(input_data)
prediction = model.predict(encoded_input)[0]

st.subheader("📈 Predicted Salary:")
st.success(f"${prediction:,.2f}")
