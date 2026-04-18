import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

st.set_page_config(page_title="Autism Prediction App", layout="centered")

st.title("🧠 Autism Prediction App")
st.markdown("""
This app predicts the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic information.
Please fill in the details below.
""")

# Input fields
st.header("Behavioral Scores (A1-A10)")
col1, col2 = st.columns(2)

a_scores = []
for i in range(1, 11):
    with col1 if i <= 5 else col2:
        score = st.selectbox(f"A{i}_Score", options=[0, 1], help=f"Score for question A{i}")
        a_scores.append(score)

st.header("Demographic Information")
col3, col4 = st.columns(2)

with col3:
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", options=encoders['gender'].classes_)
    ethnicity = st.selectbox("Ethnicity", options=encoders['ethnicity'].classes_)
    jaundice = st.selectbox("Born with Jaundice?", options=encoders['jaundice'].classes_)

with col4:
    autsim = st.selectbox("Family member with PDD?", options=encoders['autsim'].classes_)
    country = st.selectbox("Country of Residence", options=encoders['contry_of_res'].classes_)
    used_app = st.selectbox("Used screening app before?", options=encoders['used_app_before'].classes_)
    relation = st.selectbox("Relation to the person", options=encoders['relation'].classes_)

result_score = st.slider("Screening Result Score", min_value=0.0, max_value=20.0, value=5.0)

# Prediction logic
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'A1_Score': a_scores[0],
        'A2_Score': a_scores[1],
        'A3_Score': a_scores[2],
        'A4_Score': a_scores[3],
        'A5_Score': a_scores[4],
        'A6_Score': a_scores[5],
        'A7_Score': a_scores[6],
        'A8_Score': a_scores[7],
        'A9_Score': a_scores[8],
        'A10_Score': a_scores[9],
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'jaundice': jaundice,
        'autsim': autsim,
        'contry_of_res': country,
        'used_app_before': used_app,
        'result': result_score,
        'relation': relation
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical columns
    for col, encoder in encoders.items():
        df_input[col] = encoder.transform(df_input[col])
    
    # Scale numerical columns
    df_input_scaled = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(df_input_scaled)[0]
    probability = model.predict_proba(df_input_scaled)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"### Result: High Likelihood of ASD")
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.success(f"### Result: Low Likelihood of ASD")
        st.write(f"Confidence Score: {(1-probability):.2%}")

st.sidebar.info("This tool is for screening purposes only and does not replace a professional medical diagnosis.")
