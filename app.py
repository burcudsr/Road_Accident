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
    # A. Tüm sütunları 0 olan şablon oluştur
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # B. Sütun isimlerini model_columns listesindeki tam isimlerle eşleştir
    # Eğer DataFrame'de 'num_lanes' ismi varsa ata, yoksa hata alma
    def set_val(col_name, value):
        if col_name in input_df.columns:
            input_df.loc[0, col_name] = value

    set_val('num_lanes', num_lanes)
    set_val('curvature', curvature)
    set_val('speed_limit', speed_limit)
    set_val('num_reported_accidents', num_reported_accidents)
    set_val('road_signs_present', int(road_signs_present))
    set_val('public_road', int(public_road))
    set_val('holiday', int(holiday))
    set_val('school_season', int(school_season))
    
    # C. Dummy sütunları güncelle
    for col in [f'road_type_{road_type}', f'lighting_{lighting}', 
                f'weather_{weather}', f'time_of_day_{time_of_day}']:
        set_val(col, 1)
    
    # D. Tahmin
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Risk Skoru: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        st.write("Eğitimde kullanılan model sütunları:", model_columns.tolist())
