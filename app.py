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

    # Gender dengan pilihan angka dan label
    gender_map = {i: label for i, label in enumerate(encoders["gender"].classes_)}
    gender_option = st.selectbox(
        "Jenis Kelamin",
        options=list(gender_map.keys()),
        format_func=lambda x: f"{x} = {gender_map[x]}"
    )

    country_map = {i: label for i, label in enumerate(encoders["country"].classes_)}
    country_option = st.selectbox(
        "Negara",
        options=list(country_map.keys()),
        format_func=lambda x: f"{x} = {country_map[x]}"
    )

    income = st.number_input("Pendapatan", min_value=0, value=50000)
    product_quality = st.slider("Kualitas Produk", 0, 10, 5)
    service_quality = st.slider("Kualitas Layanan", 0, 10, 5)
    purchase_frequency = st.slider("Frekuensi Pembelian", 0, 10, 5)

    feedback_map = {i: label for i, label in enumerate(encoders["feedback"].classes_)}
    feedback_option = st.selectbox(
        "Feedback Score",
        options=list(feedback_map.keys()),
        format_func=lambda x: f"{x} = {feedback_map[x]}"
    )

    loyalty_map = {i: label for i, label in enumerate(encoders["loyalty"].classes_)}
    loyalty_option = st.selectbox(
        "Loyalty Level",
        options=list(loyalty_map.keys()),
        format_func=lambda x: f"{x} = {loyalty_map[x]}"
    )

    input_data = np.array([[
        age,
        gender_option,
        country_option,
        income,
        product_quality,
        service_quality,
        purchase_frequency,
        feedback_option,
        loyalty_option
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

        # Kolom wajib
        required_cols = ["Age", "Gender", "Country", "Income", "ProductQuality",
                         "ServiceQuality", "PurchaseFrequency", "FeedbackScore", "LoyaltyLevel"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"File harus memiliki kolom berikut: {missing_cols}")
        else:
            try:
                # Encoding kolom kategorikal dengan handling error bila ada label baru
                data["Gender"] = data["Gender"].map(
                    {v: k for k, v in enumerate(encoders["gender"].classes_)})
                data["Country"] = data["Country"].map(
                    {v: k for k, v in enumerate(encoders["country"].classes_)})
                data["FeedbackScore"] = data["FeedbackScore"].map(
                    {v: k for k, v in enumerate(encoders["feedback"].classes_)})
                data["LoyaltyLevel"] = data["LoyaltyLevel"].map(
                    {v: k for k, v in enumerate(encoders["loyalty"].classes_)})

                # Cek ada nilai NaN setelah mapping? berarti ada label baru yang tidak dikenal
                if data[["Gender","Country","FeedbackScore","LoyaltyLevel"]].isnull().any().any():
                    st.error("Terdapat nilai kategori di file yang tidak dikenal oleh model. Pastikan data input sesuai dengan data training.")
                else:
                    X = data[required_cols].values
                    X_scaled = scaler.transform(X)

                    if st.button("Prediksi Semua Data"):
                        preds = model.predict(X_scaled)
                        data["Prediksi Kepuasan"] = np.where(preds == 1, "PUAS", "TIDAK PUAS")
                        st.write("Hasil Prediksi:")
                        st.dataframe(data)

                        csv = data.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download hasil prediksi (CSV)",
                            data=csv,
                            file_name="hasil_prediksi.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
