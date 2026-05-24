import streamlit as st
import joblib
import pandas as pd
import os

# 1. Modelleri yükle
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'road_accident_catboost_model.joblib'))
scaler = joblib.load(os.path.join(base_path, 'road_accident_scaler.joblib'))
# model_columns, eğitimdeki orijinal sütun isimlerini içeren bir liste olmalı
model_columns = joblib.load(os.path.join(base_path, 'model_columns.joblib'))

st.title("🛣️ Road Accident Risk Prediction")

# 2. Form oluştur
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

# 3. Kesin çözüm: Tahmin Mantığı
if submitted:
    # A. Eğitimdeki sütun isimlerini kullanarak sıfırlarla dolu bir DataFrame oluştur
    # Bu, sütunların sırasını ve isimlerini garanti altına alır.
    input_df = pd.DataFrame(columns=model_columns, data=[[0.0] * len(model_columns)])
    
    # B. Kullanıcı verilerini yerleştir
    # İsimlerin model_columns içindekilerle tam aynı olduğundan emin olun
    data_map = {
        'num_lanes': float(num_lanes),
        'curvature': float(curvature),
        'speed_limit': float(speed_limit),
        'num_reported_accidents': float(num_reported_accidents),
        'road_signs_present': float(road_signs_present),
        'public_road': float(public_road),
        'holiday': float(holiday),
        'school_season': float(school_season),
        f'road_type_{road_type}': 1.0,
        f'lighting_{lighting}': 1.0,
        f'weather_{weather}': 1.0,
        f'time_of_day_{time_of_day}': 1.0
    }
    
    # C. Verileri DataFrame'e aktar (Sadece mevcut olan sütunları)
    for col, value in data_map.items():
        if col in input_df.columns:
            input_df.loc[0, col] = value
    
    # D. Ölçeklendir ve Tahmin et
    try:
        # Scaler eğitimdeki tüm sütunları bekler
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Accident Risk Score: {prediction[0]:.4f}")
    except Exception as e:
        st.error("Model hatası: " + str(e))
        st.write("Beklenen sütunlar:", list(model_columns))
