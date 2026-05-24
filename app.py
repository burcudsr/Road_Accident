import os
import streamlit as st
import joblib
import pandas as pd

# 1. Dosya yollarını belirle ve modelleri yükle
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

# 3. Hata almayan tahmin mantığı

# 3. Hata almayan tahmin mantığı
if submitted:
    # 1. Önce tamamen boş bir sözlük (dictionary) hazırla
    # Bu, tip çakışmalarını %100 engeller.
    input_data = {col: 0.0 for col in model_columns} # Her şeyi 0.0 float yap
    
    # 2. Değerleri sözlüğe ata
    input_data['num_lanes'] = float(num_lanes)
    input_data['curvature'] = float(curvature)
    input_data['speed_limit'] = float(speed_limit)
    input_data['num_reported_accidents'] = float(num_reported_accidents)
    input_data['road_signs_present'] = float(int(road_signs_present))
    input_data['public_road'] = float(int(public_road))
    input_data['holiday'] = float(int(holiday))
    input_data['school_season'] = float(int(school_season))
    
    # 3. Dummy'leri ata
    for col in [f'road_type_{road_type}', f'lighting_{lighting}', 
                f'weather_{weather}', f'time_of_day_{time_of_day}']:
        if col in input_data:
            input_data[col] = 1.0
            
    # 4. DataFrame'i sözlükten oluştur (En garantili yöntem)
    input_df = pd.DataFrame([input_data])
    
    # 5. Sütun sırasını model_columns ile zorla eşle
    input_df = input_df[model_columns]
    
    # 6. Tahmin
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Risk Skoru: {prediction[0]:.4f}")
