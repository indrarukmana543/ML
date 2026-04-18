import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# CONFIG UI
# =========================
st.set_page_config(page_title="Prediksi Produksi Tanaman", layout="centered")

st.title("🌾 Prediksi Produksi Tanaman")
st.markdown("Masukkan data untuk memprediksi hasil produksi tanaman")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model_package = joblib.load('model_produksi_tanaman.pkl')
    return model_package

model_package = load_model()
model = model_package['model']
le = model_package['label_encoder']
MODEL_EXPECTED_FEATURES = model_package['feature_names']

# =========================
# AMBIL OPSI DARI MODEL
# =========================
season_options = [
    col.replace("Season_", "").strip()
    for col in MODEL_EXPECTED_FEATURES
    if col.startswith("Season_")
]

# =========================
# INPUT USER
# =========================
st.subheader("📥 Input Data")

crop = st.selectbox("🌱 Jenis Tanaman", le.classes_)
season = st.selectbox("🌦️ Musim", season_options)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0)
    rainfall = st.number_input("Curah Hujan", min_value=0.0)
    fertilizer = st.number_input("Fertilizer", min_value=0.0)
    pesticide = st.number_input("Pesticide", min_value=0.0)

with col2:
    avg_temp = st.number_input("Suhu Rata-rata", min_value=0.0)
    max_temp = st.number_input("Suhu Maksimum", min_value=0.0)
    min_temp = st.number_input("Suhu Minimum", min_value=0.0)

# =========================
# PREPROCESSING
# =========================
def preprocess_input():
    df = pd.DataFrame([{
        'Crop': crop,
        'Season': season,
        'Area': area,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Avg_Temperature': avg_temp,
        'Max_Temperature': max_temp,
        'Min_Temperature': min_temp
    }])

    # Encode Crop
    df['Crop_Encoded'] = le.transform(df['Crop'])

    # Template sesuai model
    final_df = pd.DataFrame(0, index=df.index, columns=MODEL_EXPECTED_FEATURES)

    # Isi numerik
    for col in [
        'Area','Annual_Rainfall','Fertilizer','Pesticide',
        'Avg_Temperature','Max_Temperature','Min_Temperature'
    ]:
        if col in final_df.columns:
            final_df[col] = df[col]

    # Isi crop
    final_df['Crop_Encoded'] = df['Crop_Encoded']

    # One-hot season (AMAN dari spasi)
    for col in MODEL_EXPECTED_FEATURES:
        if col.startswith("Season_"):
            if col.replace("Season_", "").strip() == season:
                final_df[col] = 1

    return final_df

# =========================
# PREDIKSI
# =========================
st.subheader("🔍 Hasil Prediksi")

if st.button("Prediksi"):
    try:
        data = preprocess_input()
        prediction = model.predict(data)

        # Hindari nilai negatif
        prediction = np.maximum(prediction, 0)

        st.success(f"🌾 Prediksi Produksi: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")