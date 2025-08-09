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
    
    gender_map = {1: "Laki - Laki", 2: "Perempuan"}
    gender_reverse_map = {v: k for k, v in gender_map.items()}
    gender_label = st.selectbox("Jenis Kelamin", options=list(gender_map.values()))
    gender_value = gender_reverse_map[gender_label]

    country_map = {
        0: "Purwodadi",
        1: "Pulokulon",
        2: "Ngaringan",
        3: "Godong",
        4: "Gubug"
    }
    country_reverse_map = {v: k for k, v in country_map.items()}
    country_label = st.selectbox("Kecamatan", options=list(country_map.values()))
    country_value = country_reverse_map[country_label]

    income = st.number_input("Pendapatan", min_value=0, value=50000)

    # Slider label dengan 0-10, misal 0=buruk, 10=baik
    product_quality = st.slider("Kualitas Produk (0 = Buruk, 10 = Baik)", 0, 10, 5)
    service_quality = st.slider("Kualitas Layanan (0 = Buruk, 10 = Baik)", 0, 10, 5)
    purchase_frequency = st.slider("Frekuensi Pembelian (0 = Jarang, 10 = Sering)", 0, 10, 5)

    feedback_map = {0: "Tinggi", 1: "Rendah", 2: "Sedang"}
    feedback_reverse_map = {v: k for k, v in feedback_map.items()}
    feedback_label = st.selectbox("Feedback Score", options=list(feedback_map.values()))
    feedback_value = feedback_reverse_map[feedback_label]

    loyalty_map = {0: "Perunggu", 1: "Emas", 2: "Perak"}
    loyalty_reverse_map = {v: k for k, v in loyalty_map.items()}
    loyalty_label = st.selectbox("Loyalty Level", options=list(loyalty_map.values()))
    loyalty_value = loyalty_reverse_map[loyalty_label]

    # Buat input data sesuai dengan nilai numerik encoding
    input_data = np.array([[  
        age,
        gender_value,
        country_value,
        income,
        product_quality,
        service_quality,
        purchase_frequency,
        feedback_value,
        loyalty_value
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
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Data yang diupload:")
        st.dataframe(data.head())

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

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download hasil prediksi (CSV)",
                    data=csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

