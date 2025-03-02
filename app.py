import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_Outcome(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title(" diabetes Outcome Prediction")

# Input fields for each parameter
Pregnancies = st.number_input("Pregnancies",min_value=0.0, max_value=20.0,value=10.0, step=0.5)
Glucose = st.number_input("Glucose",min_value=0.0, max_value=200.0,value=100.0, step=0.5)
BloodPressure = st.number_input("BloodPressure",min_value=0.0, max_value=150.0,value=100.0, step=0.5)
SkinThickness = st.number_input("SkinThickness",min_value=0.0, max_value=100.0,value=15.0, step=0.5)
Insulin = st.number_input("Insulin",min_value=0.0, max_value=1000.0,value=79.0, step=0.5)
BMI = st.number_input("BMI",min_value=0.0, max_value=100.0,value=31.25, step=0.5)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction",min_value=0.0, max_value=5.0,value=1.0, step=0.5)
Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=50.0, step=0.1)


# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_Outcome(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Diabetes with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Did not Diabetes with probability {prob:.2f}")
