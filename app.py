if submitted:
    # 1. Kullanıcıdan alınan ham veriyi oluştur
    input_data = pd.DataFrame([{
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
    
    # 2. Get_dummies uygula
    input_encoded = pd.get_dummies(input_data)
    
    # 3. CRITICAL: Modelin eğitimde gördüğü tüm sütunları manuel tanımla 
    # (Eğitim kodundaki train_df_encoded.columns listesini buraya yazmalısın)
    # Örnek liste:
    expected_columns = [
        'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
        'road_type_highway', 'road_type_rural', 'road_type_urban',
        'lighting_daylight', 'lighting_dim', 'lighting_night',
        'weather_foggy', 'weather_rainy', 'weather_sunny',
        'road_signs_present_True', 'public_road_True',
        'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_morning',
        'holiday_True', 'school_season_True'
    ]
    
    # 4. Eksik kolonları 0 ile doldur ve sıralamayı düzelt
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_final = input_encoded[expected_columns]
    
    # 5. Scaler ve Predict
    input_scaled = scaler.transform(input_final)
    prediction = model.predict(input_scaled)
    
    # Sonuçları göster
    risk_score = prediction[0]
    st.success(f"Predicted Accident Risk Score: {risk_score:.4f}")
    
    if risk_score > 0.5:
        st.warning("⚠️ Under these conditions, the accident risk is HIGH.")
    else:
        st.info("✅ Under these conditions, the accident risk is LOW.")
