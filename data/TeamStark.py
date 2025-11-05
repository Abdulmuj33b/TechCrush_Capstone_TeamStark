import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the path to the saved model file (adjust if necessary)
# In Streamlit deployed apps, you might need to adjust paths or load from cloud storage.
# For Colab, this path assumes the file is in your mounted Drive.
model_path = '/content/drive/MyDrive/heart-disease-project/models/final_xgb_optuna.pkl'

# Load the model pipeline
@st.cache_resource # Cache the model to avoid reloading on each rerun
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure the path is correct and the file exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model(model_path)

st.title("TeamHeart Disease Prediction App")
st.write("Enter patient details to get a heart disease risk prediction.")

if model is None:
    st.stop() # Stop the app if model loading failed

# Define input widgets based on the features
# Referencing the feature list from the prediction function:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=50)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3]) # Assuming 4 types
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.sidebar.number_input("Serum Cholestoral (chol)", min_value=0, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2]) # Assuming 3 types
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2]) # Assuming 3 types
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4]) # Assuming 0-4
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3]) # Assuming 4 types

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Make prediction
if st.sidebar.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1] # Probability of class 1
    probability_of_disease = prediction_proba[0] * 100

    st.subheader("Prediction Result")
    if probability_of_disease >= 50: # You can adjust this threshold
        st.error(f"Predicted probability of heart disease: **{probability_of_disease:.2f}%**")
        st.write("Based on the inputs, there is a high likelihood of heart disease.")
    else:
        st.success(f"Predicted probability of heart disease: **{probability_of_disease:.2f}%**")
        st.write("Based on the inputs, there is a low likelihood of heart disease.")

    st.write("Note: This is a model prediction and should not replace professional medical advice.")
