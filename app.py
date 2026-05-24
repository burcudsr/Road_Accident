import os
import streamlit as st
import joblib
import pandas as pd

# Dosyaların bulunduğu klasörü otomatik olarak belirle
base_path = os.path.dirname(__file__)

# Dosya yollarını bu klasöre göre birleştir
model_path = os.path.join(base_path, 'road_accident_catboost_model.joblib')
scaler_path = os.path.join(base_path, 'road_accident_scaler.joblib')
columns_path = os.path.join(base_path, 'model_columns.joblib')

# Modelleri bu yolları kullanarak yükle
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(columns_path)

st.title("🛣️ Road Accident Risk Prediction")

# 2. User Input Form
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

# 3. Prediction Process
if submitted:
    # 1. Kullanıcıdan gelen veriyi bir dict olarak al ve DataFrame yap
    input_data = {
        'road_type': [road_type],
        'num_lanes': [num_lanes],
        'curvature': [curvature],
        'speed_limit': [speed_limit],
        'lighting': [lighting],
        'weather': [weather],
        'time_of_day': [time_of_day],
        'num_reported_accidents': [num_reported_accidents],
        'road_signs_present': [road_signs_present],
        'public_road': [public_road],
        'holiday': [holiday],
        'school_season': [school_season]
    }
    input_df = pd.DataFrame(input_data)
    
    # 2. Eğitimdeki aynı mantıkla get_dummies uygula (sadece object sütunları)
    # Eğitimde cat_cols sadece object'ti, burada da öyle yapıyoruz
    input_encoded = pd.get_dummies(input_df, columns=['road_type', 'lighting', 'weather', 'time_of_day'])
    
    # 3. Bool kolonları int yap
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        input_encoded[col] = input_encoded[col].astype(int)
        
    # 4. KRİTİK ADIM: Reindex
    # Modelin eğitimde gördüğü tüm sütunları zorla oluştur, eksikleri 0 ile doldur
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 5. Scaler ve Predict
    input_scaled = scaler.transform(input_final)
    prediction = model.predict(input_scaled)
    
    # 5. Display Results
    risk_score = prediction[0]
    st.success(f"Predicted Accident Risk Score: {risk_score:.4f}")
    
    if risk_score > 0.5:
        st.warning("⚠️ Under these conditions, the accident risk is HIGH.")
    else:
        st.info("✅ Under these conditions, the accident risk is LOW.")
