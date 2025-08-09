import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load dataset (hanya untuk opsi kategori)
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

menu = st.selectbox("Pilih Mode Prediksi", ["Manual", "Upload File (CSV/XLSX)"])

if menu == "Manual":
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

elif menu == "Upload File (CSV/XLSX)":
    uploaded_file = st.file_uploader("Upload file CSV atau XLSX", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Baca file sesuai ekstensi
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Data yang diupload:")
        st.dataframe(data.head())

        # Pastikan kolom ada
        required_cols = ["Age", "Gender", "Country", "Income", "ProductQuality", 
                         "ServiceQuality", "PurchaseFrequency", "FeedbackScore", "LoyaltyLevel"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"File harus memiliki kolom berikut: {missing_cols}")
        else:
            # Encoding kolom kategorikal
            data["Gender"] = encoders["gender"].transform(data["Gender"])
            data["Country"] = encoders["country"].transform(data["Country"])
            data["FeedbackScore"] = encoders["feedback"].transform(data["FeedbackScore"])
            data["LoyaltyLevel"] = encoders["loyalty"].transform(data["LoyaltyLevel"])

            X = data[required_cols].values
            X_scaled = scaler.transform(X)

            if st.button("Prediksi Semua Data"):
                preds = model.predict(X_scaled)
                data["Prediksi Kepuasan"] = np.where(preds==1, "PUAS", "TIDAK PUAS")
                st.write("Hasil Prediksi:")
                st.dataframe(data)

                # Opsi download hasil prediksi
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download hasil prediksi (CSV)",
                    data=csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
