import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

import base64

# Function to convert image to base64
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Mental_health_in_Tech.png")
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{logo_base64}' width='120'/>
    </div>
    <div style='height: 40px;'></div>
    """,
    unsafe_allow_html=True
)

# Load model, scaler, encoders
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Mental Health Risk Predictor for Tech Workers")

st.markdown("### Answer the following to see your predicted mental health risk:")

# Input form
age = st.slider("Age", 18, 65, 30)
gender_input = st.selectbox("Gender", ['Male', 'Female', 'Other'])
country = st.selectbox("Country", ['United States', 'India', 'Canada', 'Germany', 'United Kingdom', 'Other'])
family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No'])
care_options = st.selectbox("Do you get access to care options?", ['Yes', 'No', 'Not sure'])
wellness_program = st.selectbox("Are employer wellness programs held?", ['Yes', 'No'])
seek_help = st.selectbox("Would you be encouraged to seek help?", ['Yes', 'No'])
remote_work = st.selectbox("Do you work remotely?", ['Yes', 'No'])
work_interfere = st.selectbox("Does your mental health interfere with work?", ['Never', 'Rarely', 'Sometimes', 'Often'])
interview = st.selectbox("Would you be comfortable discussing your mental health in an interview?", ['Yes', 'No', 'Maybe'])

gender = gender_input.lower()
if 'male' in gender:
    gender = 'Male'
elif 'female' in gender:
    gender = 'Female'
else:
    gender = 'Other'

input_dict = {
    'Age': age,
    'Gender': gender,
    'Country': country,
    'family_history': family_history,
    'benefits': benefits,
    'care_options': care_options,
    'wellness_program': wellness_program,
    'seek_help': seek_help,
    'remote_work': remote_work,
    'work_interfere': work_interfere,
    'mental_health_interview': interview
}
input_df = pd.DataFrame([input_dict])

for col in input_df.select_dtypes(include='object').columns:
    encoder = encoders[col]
    value = input_df[col].iloc[0]
    if value not in encoder.classes_:
        st.error(f"Invalid input for {col}: '{value}' not seen in training. Please try a different option.")
        st.stop()
    input_df[col] = encoder.transform([value])

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ At Risk of Mental Health Issues! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Not at Risk. Probability: {prob:.2f}")
