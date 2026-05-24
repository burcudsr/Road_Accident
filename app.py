import os
import streamlit as st
import joblib
import pandas as pd

# 1. Modelleri ve sütun listesini yükle
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

# 3. Kesin Çözüm: Tahmin Mantığı
if submitted:
    # 1. Eğitimde modelin gördüğü tam sütun listesini baz alarak bir boş sözlük oluştur
    # Bu, scaler'ın beklediği tüm sütun isimlerini içerir.
    input_data = {col: 0.0 for col in model_columns}
    
    # 2. Sayısal ve boolean değerleri sözlüğe (dictionary) ata
    input_data['num_lanes'] = float(num_lanes)
    input_data['curvature'] = float(curvature)
    input_data['speed_limit'] = float(speed_limit)
    input_data['num_reported_accidents'] = float(num_reported_accidents)
    
    # Checkbox değerlerini 1.0 veya 0.0 olarak ata
    # İsimlerin model_columns'taki ile birebir aynı olduğundan emin olun (suffix'lere dikkat!)
    # Hata mesajınıza göre 'holiday_True' gibi isimler bekliyor olabilir.
    if 'holiday_True' in input_data: input_data['holiday_True'] = 1.0 if holiday else 0.0
    if 'public_road_True' in input_data: input_data['public_road_True'] = 1.0 if public_road else 0.0
    if 'road_signs_present_True' in input_data: input_data['road_signs_present_True'] = 1.0 if road_signs_present else 0.0
    if 'school_season_True' in input_data: input_data['school_season_True'] = 1.0 if school_season else 0.0
    
    # 3. Kategorik (dummy) değişkenleri ata
    for col in [f'road_type_{road_type}', f'lighting_{lighting}', 
                f'weather_{weather}', f'time_of_day_{time_of_day}']:
        if col in input_data:
            input_data[col] = 1.0
            
    # 4. DataFrame'i sözlükten oluştur (En güvenli yöntem)
    input_df = pd.DataFrame([input_data])
    
    # 5. Sütun sırasını model_columns ile zorla eşle
    input_df = input_df[model_columns]
    
    # 6. Tahmin yap
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Accident Risk Score: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Tahmin Hatası: {str(e)}")
        # Hata devam ederse, beklentiyi görelim:
        st.write("Modelin beklediği sütunlar (model_columns):", model_columns)
