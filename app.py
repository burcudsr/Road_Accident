import os
import streamlit as st
import joblib
import pandas as pd

# 1. Modelleri yükle
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'road_accident_catboost_model.joblib'))
scaler = joblib.load(os.path.join(base_path, 'road_accident_scaler.joblib'))
model_columns = joblib.load(os.path.join(base_path, 'model_columns.joblib'))

st.title("🛣️ Road Accident Risk Prediction")

# 2. Arayüz
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
        num_reported_accidents = st.number_input("Reported Accidents", 0, 7, 0)
        road_signs_present = st.checkbox("Road Signs Present?")
        public_road = st.checkbox("Public Road?")
        holiday = st.checkbox("Holiday?")
        school_season = st.checkbox("School Season?")
    submitted = st.form_submit_button("Predict Risk")

# 3. Final Tahmin Mantığı
if submitted:
    # A. Tüm sütunları 0.0 float olarak hazırla
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # B. Sayısal verileri yerleştir
    input_df.loc[0, 'num_lanes'] = float(num_lanes)
    input_df.loc[0, 'curvature'] = float(curvature)
    input_df.loc[0, 'speed_limit'] = float(speed_limit)
    input_df.loc[0, 'num_reported_accidents'] = float(num_reported_accidents)
    
    # C. Checkbox verilerini yerleştir
    input_df.loc[0, 'road_signs_present'] = 1.0 if road_signs_present else 0.0
    input_df.loc[0, 'public_road'] = 1.0 if public_road else 0.0
    input_df.loc[0, 'holiday'] = 1.0 if holiday else 0.0
    input_df.loc[0, 'school_season'] = 1.0 if school_season else 0.0
    
    # D. Kategorik verileri yerleştir
    for cat in [road_type, lighting, weather, time_of_day]:
        col_name = f"road_type_{cat}" if cat in ["urban", "rural"] else \
                   f"lighting_{cat}" if cat in ["dim", "night"] else \
                   f"weather_{cat}" if cat in ["foggy", "rainy"] else \
                   f"time_of_day_{cat}" if cat in ["evening", "morning"] else None
        
        if col_name and col_name in input_df.columns:
            input_df.loc[0, col_name] = 1.0
            
    # E. Tahmin
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Accident Risk Score: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Tahmin Hatası: {e}")
