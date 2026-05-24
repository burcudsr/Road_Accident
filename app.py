import streamlit as st
import pandas as pd
import joblib

# 1. Loading Libraries and Models
# Ensure your model files are in the same directory as the script
try:
    model = joblib.load('road_accident_catboost_model.joblib')
    scaler = joblib.load('road_accident_scaler.joblib')
    # model_columns should contain the list of dummy-encoded column names from training
    model_columns = joblib.load('model_columns.joblib') 
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.title("🛣️ Road Accident Risk Prediction Panel")

# 2. User Interface
with st.form("risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        road_type = st.selectbox("Road Type", ["urban", "rural", "highway"])
        num_lanes = st.slider("Number of Lanes", 1, 4, 2)
        curvature = st.slider("Curvature (0-1)", 0.0, 1.0, 0.5)
        speed_limit = st.number_input("Speed Limit", 25, 70, 45)
        lighting = st.selectbox("Lighting", ["daylight", "dim", "night"])
        weather = st.selectbox("Weather", ["sunny", "rainy", "foggy"])
        
    with col2:
        time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"])
        num_reported_accidents = st.number_input("Number of Reported Accidents", 0, 7, 0)
        road_signs_present = st.checkbox("Are Road Signs Present?")
        public_road = st.checkbox("Is it a Public Road?")
        holiday = st.checkbox("Is it a Holiday?")
        school_season = st.checkbox("Is it School Season?")

    submitted = st.form_submit_button("Predict Risk")

# 3. Prediction Process
if submitted:
    # Convert user input to DataFrame
    input_df = pd.DataFrame([{
        'road_type': road_type,
        'num_lanes': num_lanes,
        'curvature': curvature,
        'speed_limit': speed_limit,
        'lighting': lighting,
        'weather': weather,
        'road_signs_present': road_signs_present,
        'public_road': public_road,
        'time_of_day': time_of_day,
        'holiday': holiday,
        'school_season': school_season,
        'num_reported_accidents': num_reported_accidents
    }])
    
    # Feature Engineering (One-Hot Encoding)
    input_encoded = pd.get_dummies(input_df)
    
    # Align with training columns (add missing columns as 0)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scaling
    input_scaled = scaler.transform(input_final)
    
    # Prediction
    prediction = model.predict(input_scaled)
    
    # Results
    risk_score = prediction[0]
    st.success(f"Predicted Accident Risk Score: {risk_score:.4f}")
    
    if risk_score > 0.5:
        st.warning("⚠️ Under these conditions, the accident risk is HIGH.")
    else:
        st.info("✅ Under these conditions, the accident risk is LOW.")
