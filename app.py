import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load dataset (hanya untuk mengambil opsi kategori)
@st.cache_data
def load_data():
    return pd.read_csv("customer_feedback.csv")

# Load model, scaler, encoder
@st.cache_resource
def load_model():
    model = pickle.load(open("model_knn.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, scaler, encoders

df = load_data()
model, scaler, encoders = load_model()

st.title("🤖 Prediksi Kepuasan Pelanggan")

age = st.number_input("Usia", min_value=18, max_value=100, value=30)
gender = st.selectbox("Jenis Kelamin", encoders["gender"].classes_)
country = st.selectbox("Negara", encoders["country"].classes_)
income = st.number_input("Pendapatan", min_value=0, value=50000)
product_quality = st.slider("Kualitas Produk", 0, 10, 5)
service_quality = st.slider("Kualitas Layanan", 0, 10, 5)
purchase_frequency = st.slider("Frekuensi Pembelian", 0, 10, 5)
feedback_score = st.selectbox("Feedback Score", encoders["feedback"].classes_)
loyalty_level = st.selectbox("Loyalty Level", encoders["loyalty"].classes_)

input_data = np.array([[  
    age,
    encoders["gender"].transform([gender])[0],
    encoders["country"].transform([country])[0],
    income,
    product_quality,
    service_quality,
    purchase_frequency,
    encoders["feedback"].transform([feedback_score])[0],
    encoders["loyalty"].transform([loyalty_level])[0]
]])

input_data_scaled = scaler.transform(input_data)

if st.button("Prediksi"):
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 1:
        st.success("✅ Pelanggan PUAS")
    else:
        st.error("⚠️ Pelanggan TIDAK PUAS")
