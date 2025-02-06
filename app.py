import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3

# Load models
crop_model = joblib.load("crop_recommendation_model.pkl")
fertilizer_model = joblib.load("fertilizer_recommendation_model.pkl")

# Crop-soil mapping
crop_soil_map = {
    'rice': ['Clayey', 'Loamy'], 'maize': ['Loamy', 'Sandy'], 'chickpea': ['Loamy', 'Red'],
    'kidneybeans': ['Loamy', 'Red'], 'pigeonpeas': ['Loamy', 'Black'], 'mothbeans': ['Sandy', 'Red'],
    'mungbean': ['Loamy', 'Black'], 'blackgram': ['Loamy', 'Red'], 'lentil': ['Loamy', 'Black'],
    'pomegranate': ['Loamy', 'Sandy'], 'banana': ['Loamy', 'Clayey'], 'mango': ['Loamy', 'Clayey'],
    'grapes': ['Loamy', 'Sandy'], 'watermelon': ['Sandy', 'Loamy'], 'muskmelon': ['Sandy', 'Loamy'],
    'apple': ['Loamy', 'Clayey'], 'orange': ['Loamy', 'Sandy'], 'papaya': ['Loamy', 'Clayey'],
    'coconut': ['Loamy', 'Sandy'], 'cotton': ['Loamy', 'Sandy'], 'jute': ['Loamy', 'Clayey'],
    'coffee': ['Loamy', 'Red']
}

# Streamlit UI
st.title("Agriculture Crop and Fertilizer Recommendation")

# Input fields for crop recommendation
st.subheader("Crop Recommendation")
nitrogen = st.number_input("Enter Nitrogen (N)", min_value=0.0, max_value=200.0, step=1.0)
phosphorus = st.number_input("Enter Phosphorus (P)", min_value=0.0, max_value=200.0, step=1.0)
potassium = st.number_input("Enter Potassium (K)", min_value=0.0, max_value=200.0, step=1.0)
temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
ph = st.number_input("Enter pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=5000.0, step=1.0)

if st.button("Get Crop Recommendation"):
    # Predict crop
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    probabilities = crop_model.predict_proba(input_data)
    top_indices = np.argsort(probabilities[0])[-3:][::-1]
    top_crops = [crop_model.classes_[i] for i in top_indices]
    
    # Show recommended crops
    st.write(f"Top 3 recommended crops: {', '.join(top_crops)}")

    # Store recommendations in session state
    st.session_state.top_crops = top_crops
    st.session_state.crop_selected = None

# If crops are recommended, let the user select one
if "top_crops" in st.session_state and st.session_state.top_crops:
    chosen_crop = st.selectbox("Select a crop", st.session_state.top_crops)
    if chosen_crop:
        st.session_state.crop_selected = chosen_crop

# Fertilizer recommendation based on chosen crop
if "crop_selected" in st.session_state and st.session_state.crop_selected:
    st.subheader("Fertilizer Recommendation")
    
    # Fetching soil type and moisture
    soil_type = crop_soil_map.get(st.session_state.crop_selected, ["Loamy"])[0]
    
    # Asking for moisture and soil type from the user
    moisture = st.number_input("Enter Moisture (%)", min_value=0.0, max_value=100.0, step=1.0)
    soil_type = st.selectbox(f"Select soil type for {st.session_state.crop_selected}", crop_soil_map[st.session_state.crop_selected], index=0)

    if st.button("Get Fertilizer Recommendation"):
        # Generate input for fertilizer recommendation
        fertilizer_input_data = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Soil_Type': soil_type,
            'Crop_Type': st.session_state.crop_selected
        }
        
        # Create a dataframe for prediction
        fertilizer_input_df = pd.DataFrame([fertilizer_input_data])
        fertilizer_input_df = pd.get_dummies(fertilizer_input_df, columns=['Soil_Type', 'Crop_Type'])

        # Load fertilizer dataset to align feature columns
        fertilizer_df = pd.read_csv("Fertilizer Recommendation.csv")
        X_fert = pd.get_dummies(fertilizer_df[['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorus', 'Soil_Type', 'Crop_Type']])

        # Align columns between input and model
        for col in X_fert.columns:
            if col not in fertilizer_input_df.columns:
                fertilizer_input_df[col] = 0
        fertilizer_input_df = fertilizer_input_df[X_fert.columns]
        
        # Predict fertilizer
        fertilizer = fertilizer_model.predict(fertilizer_input_df)[0]
        st.write(f"Recommended Fertilizer for {st.session_state.crop_selected}: {fertilizer}") 