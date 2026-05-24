import os
import streamlit as st
import joblib
import pandas as pd

# 1. Dosya yollarını belirle ve modelleri yükle
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'road_accident_catboost_model.joblib')
scaler_path = os.path.join(base_path, 'road_accident_scaler.joblib')
columns_path = os.path.join(base_path, 'model_columns.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(columns_path)

st.title("🛣️ Road Accident Risk Prediction")

# 2. Arayüz (Form)
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

# 3. Tahmin İşlemi
if submitted:
    # A. Modelin beklediği tüm sütunları 0 (int) olarak içeren boş bir şablon oluştur
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # B. Sayısal değişkenleri ata
    input_df['num_lanes'] = num_lanes
    input_df['curvature'] = curvature
    input_df['speed_limit'] = speed_limit
    input_df['num_reported_accidents'] = num_reported_accidents
    
    # C. Kategorik değişkenler (drop_first=True mantığı ile)
    # Eğer değer 'ilk' (eğitimde düşürülen) kategori değilse sütunu 1 yap
    if road_type != 'highway':
        input_df[f'road_type_{road_type}'] = 1
    if lighting != 'daylight':
        input_df[f'lighting_{lighting}'] = 1
    if weather != 'sunny':
        input_df[f'weather_{weather}'] = 1
    if time_of_day != 'morning':
        input_df[f'time_of_day_{time_of_day}'] = 1
    
    # D. Boolean değişkenleri int (1 veya 0) formatında ata
    input_df['road_signs_present'] = int(road_signs_present)
    input_df['public_road'] = int(public_road)
    input_df['holiday'] = int(holiday)
    input_df['school_season'] = int(school_season)
    
    # E. Ölçeklendirme ve Tahmin
    # Şimdi sütunlar ve tipler modelin beklentisiyle %100 uyumlu
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    # F. Sonuçları Göster
    risk_score = prediction[0]
    st.success(f"Predicted Accident Risk Score: {risk_score:.4f}")
    
    if risk_score > 0.5:
        st.warning("⚠️ Under these conditions, the accident risk is HIGH.")
    else:
        st.info("✅ Under these conditions, the accident risk is LOW.")
