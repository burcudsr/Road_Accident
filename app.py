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

# 3. Tahmin Mantığı
if submitted:
    # A. Eğitimdeki tüm sütunları 0 olarak içeren boş bir şablon oluştur
    # Bu adım Scaler'ın "sütun isimleri uyuşmuyor" hatasını kesin olarak keser
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # B. Sayısal ve boolean değerleri şablona ata
    input_df.loc[0, 'num_lanes'] = int(num_lanes)
    input_df.loc[0, 'curvature'] = float(curvature)
    input_df.loc[0, 'speed_limit'] = int(speed_limit)
    input_df.loc[0, 'num_reported_accidents'] = int(num_reported_accidents)
    input_df.loc[0, 'road_signs_present'] = int(road_signs_present)
    input_df.loc[0, 'public_road'] = int(public_road)
    input_df.loc[0, 'holiday'] = int(holiday)
    input_df.loc[0, 'school_season'] = int(school_season)
    
    # C. Kategorik (dummy) değişkenleri ata
    # Modelin bildiği sütunları tek tek kontrol et
    category_cols = [
        f'road_type_{road_type}', 
        f'lighting_{lighting}', 
        f'weather_{weather}', 
        f'time_of_day_{time_of_day}'
    ]
    
    for col in category_cols:
        if col in input_df.columns:
            input_df.loc[0, col] = 1
            
    # D. Ölçeklendirme ve Tahmin
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        # E. Sonuç
        st.success(f"Predicted Accident Risk Score: {prediction[0]:.4f}")
        
        if prediction[0] > 0.5:
            st.warning("⚠️ High accident risk detected.")
        else:
            st.info("✅ Low accident risk detected.")
            
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
