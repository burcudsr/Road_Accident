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

# 3. Hata ayıklamalı tahmin bloğu
if submitted:
    # 1. Başlangıç şablonu
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # 2. Değer atamaları
    input_df.loc[0, 'num_lanes'] = float(num_lanes)
    input_df.loc[0, 'curvature'] = float(curvature)
    input_df.loc[0, 'speed_limit'] = float(speed_limit)
    input_df.loc[0, 'num_reported_accidents'] = float(num_reported_accidents)
    
    # Checkbox'ları modelde varsa atıyoruz
    for col in ['holiday', 'public_road', 'road_signs_present', 'school_season']:
        if col in input_df.columns:
            input_df.loc[0, col] = 1.0 if (locals()[col]) else 0.0
            
    # 3. Kategorikler
    for col in [f'road_type_{road_type}', f'lighting_{lighting}', 
                f'weather_{weather}', f'time_of_day_{time_of_day}']:
        if col in input_df.columns:
            input_df.loc[0, col] = 1.0

    # --- DEBUG PANELİ (Modelin ne beklediğini burada göreceksin) ---
    st.write("### Hata Ayıklama Paneli")
    st.write("Modelin Beklediği Sütun Sayısı:", len(model_columns))
    st.write("Modelin Beklediği Sütunlar:", list(model_columns))
    st.write("Senin Oluşturduğun Sütunlar:", list(input_df.columns))
    st.write("DataFrame Önizleme (Sadece 1 olanlar):", input_df.loc[:, (input_df != 0).any(axis=0)])
    # -----------------------------------------------------------

    # 4. Tahmin
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Risk Skoru: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Tahmin Hatası: {e}")
